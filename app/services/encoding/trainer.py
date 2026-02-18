import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from app.services.encoding.unknown_detector import UnknownDetector


class Trainer:
    def __init__(self, model: nn.Module, config: dict, class_weights: Optional[torch.Tensor] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_config = config["training"]
        self.data_config = config["data"]
        self.unk_config = config["unknown_detection"]

        # Loss function - weighted when class imbalance handling is active
        if self.data_config.get("use_class_weights") and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Unknown detector (calibrated on val set after training)
        self.unknown_detector = UnknownDetector(self.unk_config) if self.unk_config.get("enabled") else None

        # State
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
        }

    def train(self, train_loader, val_loader):
        """
        Run the full training loop for all epochs, then calibrate the unknown detector.
        Saves a checkpoint whenever the monitored metric improves.
        self.train_loader must be set before calling this method.

        :param train_loader: DataLoader for the training set
        :param val_loader: DataLoader for the validation set, also used for unknown detector calibration
        """
        self._setup_optimizer(train_loader)
        cfg = self.train_config

        print(f"\n{'=' * 60}\n  TRAINING  ({cfg['num_epochs']} epochs, device={self.device})\n{'=' * 60}")

        for epoch in range(cfg["num_epochs"]):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate(val_loader)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_weighted_f1"].append(val_metrics["weighted_f1"])

            print(
                f"  Epoch {epoch + 1:>2}/{cfg['num_epochs']} | "
                f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
                f"macro_f1={val_metrics['macro_f1']:.4f}  acc={val_metrics['accuracy']:.4f}"
            )

            # Save best model
            metric_val = val_metrics[cfg["monitor_metric"]]
            if metric_val > self.best_metric:
                self.best_metric = metric_val
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
                print(f"    Best {cfg['monitor_metric']} → {metric_val:.4f} (saved)")

        print(f"\n  Best {cfg['monitor_metric']}: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")

        # Calibrate unknown detector on val set
        if self.unknown_detector is not None:
            print("\n  Calibrating unknown detector on validation set...")
            self.unknown_detector.calibrate(val_loader, self.device, self.model)

    def evaluate(self, dataloader, return_predictions: bool = False, label_mapping: Optional[dict] = None):
        """
        Evaluate the model on a dataloader with sliding window aggregation.
        Windows belonging to the same document are merged into a single prediction
        using the pooling strategy defined in the data config.

        :param dataloader: DataLoader to evaluate on (val or test)
        :param return_predictions: If True, also returns raw prediction and label arrays alongside metrics
        :param label_mapping: If provided, prints a full per-class classification report using original label names
        :return: Dict with loss, accuracy, macro_f1 and weighted_f1. If return_predictions=True, also returns (preds, labels) arrays
        """
        self.model.eval()

        all_logits, all_labels, all_doc_ids = [], [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                total_loss += self.criterion(logits, labels).item()

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_doc_ids.append(batch["doc_id"].numpy())

        # Aggregate windows to one prediction per document
        doc_logits, doc_labels = self._aggregate_windows(
            torch.cat(all_logits),
            torch.cat(all_labels),
            np.concatenate(all_doc_ids),
        )

        preds = torch.argmax(doc_logits, dim=1).numpy()
        labels_np = doc_labels.numpy()

        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": float((preds == labels_np).mean()),
            "macro_f1": f1_score(labels_np, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels_np, preds, average="weighted", zero_division=0),
        }

        if label_mapping:
            target_names = [label_mapping[i] for i in sorted(label_mapping)]
            print("\n" + classification_report(labels_np, preds, target_names=target_names, digits=4))

        if return_predictions:
            return metrics, preds, labels_np
        return metrics

    def plot_history(self):
        """
        Plot and save training curves (loss, macro F1, accuracy) to the checkpoint directory.
        Called after training completes.
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(self.history["train_loss"], label="train")
        axes[0].plot(self.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.history["val_macro_f1"], color="orange")
        axes[1].set_title("Val Macro F1")
        axes[1].grid(alpha=0.3)

        axes[2].plot(self.history["val_accuracy"], color="green")
        axes[2].set_title("Val Accuracy")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.train_config["checkpoint_dir"], "training_history.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  History saved → {path}")

    def load_best(self):
        """
        Load the best saved checkpoint weights into the model.
        Must be called before evaluating on the test set to ensure
        the best epoch's weights are used rather than the last epoch's.
        """
        path = os.path.join(self.train_config["checkpoint_dir"], "best_model.pt")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded best model from epoch {state['epoch'] + 1}")

    def _setup_optimizer(self, train_loader):
        """
        Initialize AdamW optimizer and linear warmup scheduler.
        Weight decay is applied only to non-bias and non-LayerNorm parameters
        as decaying these can harm training stability.

        :param train_loader: Training DataLoader, used to compute the total number of training steps
        """
        cfg = self.train_config

        # Weight decay only on non-bias/LayerNorm params
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(params, lr=cfg["learning_rate"], eps=1e-8)

        total_steps = len(train_loader) * cfg["num_epochs"] // cfg["gradient_accumulation_steps"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        print(f"  Optimizer ready — total steps: {total_steps}, warmup: {warmup_steps}")

    def _train_epoch(self, epoch: int) -> float:
        """
        Run one full training epoch with gradient accumulation.
        Centroids are updated every batch so they are ready for unknown detection calibration.

        :param epoch: Current epoch index, used only for the progress bar label
        :return: Average training loss for this epoch
        """
        self.model.train()
        cfg = self.train_config
        total_loss = 0.0
        self.optimizer.zero_grad()

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                   desc=f"Epoch {epoch + 1}", leave=False)

        for step, batch in bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, embeddings = self.model(input_ids, attention_mask, return_embeddings=True)
            loss = self.criterion(logits, labels) / cfg["gradient_accumulation_steps"]
            loss.backward()

            # Update centroids for unknown detection
            self.model.update_centroids(embeddings.detach(), labels)

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * cfg["gradient_accumulation_steps"]
            bar.set_postfix(loss=f"{loss.item() * cfg['gradient_accumulation_steps']:.4f}")

        return total_loss / len(self.train_loader)

    def _aggregate_windows(self, logits: torch.Tensor, labels: torch.Tensor,
                           doc_ids: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge per-window logits into a single prediction per document.
        The aggregation strategy is read from data config (window_pooling).

        Strategies:
            max   — take the maximum probability per class across all windows, most confident window wins
            mean  — average the raw logits across all windows, smoother result
            first — use only the first window, fastest but ignores rest of the document

        :param logits: Logits for every window across all documents [total_windows, num_classes]
        :param labels: Label for every window, all windows of the same document share the same label [total_windows]
        :param doc_ids: Document index for every window, used to group windows back into documents [total_windows]
        :return: Aggregated logits and labels, one entry per document
        """
        pooling = self.data_config.get("window_pooling", "mean")

        agg_logits, agg_labels = [], []
        for doc_id in np.unique(doc_ids):
            mask = doc_ids == doc_id
            doc_logits = logits[mask]
            doc_label = labels[mask][0]  # all windows share the same label

            if pooling == "max":
                probs = torch.softmax(doc_logits, dim=1)
                agg = torch.log(torch.max(probs, dim=0).values + 1e-10)
            elif pooling == "mean":
                agg = doc_logits.mean(dim=0)
            elif pooling == "first":
                agg = doc_logits[0]
            else:
                raise ValueError(f"Unknown window_pooling: {pooling}")

            agg_logits.append(agg)
            agg_labels.append(doc_label)

        return torch.stack(agg_logits), torch.stack(agg_labels)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model weights and training state to disk.

        :param epoch: Current epoch index, used in the filename for non-best checkpoints
        :param is_best: If True, saves as best_model.pt, overwriting the previous best
        """
        os.makedirs(self.train_config["checkpoint_dir"], exist_ok=True)
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.train_config["checkpoint_dir"], filename)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
        }

        # Save unknown detector threshold if it was calibrated
        if self.unknown_detector is not None and self.unknown_detector.distance_threshold is not None:
            checkpoint["unknown_detector_threshold"] = self.unknown_detector.distance_threshold

        torch.save(checkpoint, path)