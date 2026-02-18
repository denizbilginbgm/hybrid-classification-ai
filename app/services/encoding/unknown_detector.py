from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm


class UnknownDetector:
    def __init__(self, config: dict):
        self.config = config
        self.distance_threshold: Optional[float] = None

    def calibrate(self, val_loader, device, model):
        """
        Compute and store the distance threshold using the validation set.
        Only correctly predicted samples are used to establish what "normal" distances look like
        (misclassified samples would set the boundary too loosely).

        The threshold is set at the Nth percentile of observed distances,
        where N is distance_threshold_percentile from config.

        :param val_loader: DataLoader for the validation set
        :param device: torch device the model is running on
        :param model: Trained model with class_centroids buffer populated during training
        """
        model.eval()
        distances = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Calibrating", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, embeddings = model(input_ids, attention_mask, return_embeddings=True)
                preds = torch.argmax(logits, dim=1)

                # Only use correctly predicted samples to set the "normal" boundary
                for emb, label in zip(embeddings[preds == labels], labels[preds == labels]):
                    dist = torch.norm(emb - model.class_centroids[label]).item()
                    distances.append(dist)

        self.distance_threshold = float(np.percentile(distances, self.config["distance_threshold_percentile"]))
        print(f"  Distance threshold ({self.config['distance_threshold_percentile']}th pct): "
              f"{self.distance_threshold:.4f}")

    def predict(self, logits: torch.Tensor, embeddings: torch.Tensor, model) -> torch.Tensor:
        """
        Determine which samples in a batch are likely unknown/out-of-distribution.
        Runs the enabled strategies and combines their results using the configured logic.

        Must be called after calibrate(), otherwise distance_threshold is None
        and the centroid distance strategy will be skipped silently.

        :param logits: Raw model output before softmax [batch_size, num_classes]
        :param embeddings: [CLS] embeddings from the encoder, before the classification head [batch_size, hidden_size]
        :param model: Model instance with class_centroids buffer, used to look up per-class centers
        :return: Boolean mask where True indicates the sample is likely unknown [batch_size]
        """
        batch = logits.size(0)
        device = logits.device

        # Strategy 1: confidence
        if self.config.get("use_confidence_threshold"):
            probs = torch.softmax(logits, dim=1)
            low_conf = probs.max(dim=1).values < self.config["confidence_threshold"]
        else:
            low_conf = torch.zeros(batch, dtype=torch.bool, device=device)

        # Strategy 2: centroid distance
        if self.config.get("use_centroid_distance") and self.distance_threshold is not None:
            pred_classes = logits.argmax(dim=1)
            far_from_centroid = torch.zeros(batch, dtype=torch.bool, device=device)
            for i in range(batch):
                dist = torch.norm(embeddings[i] - model.class_centroids[pred_classes[i]]).item()
                far_from_centroid[i] = dist > self.distance_threshold
        else:
            far_from_centroid = torch.zeros(batch, dtype=torch.bool, device=device)

        # Combination
        # OR: either condition is enough to flag as unknown (more sensitive, more false positives)
        # AND: both conditions must hold to flag as unknown (more conservative, fewer false positives)
        if self.config.get("unknown_logic", "OR") == "OR":
            return low_conf | far_from_centroid
        return low_conf & far_from_centroid