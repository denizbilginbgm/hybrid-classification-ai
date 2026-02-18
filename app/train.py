import importlib
import json
import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from app.core import config
from app.services.datasets.data_preprocessor import DataPreprocessor
from app.services.datasets.windowed_dataset_creator import WindowedDatasetCreator
from app.services.encoding.trainer import Trainer
from app.utils.utils import load_model_class

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def resolve_checkpoint_dir(config_path: str) -> str:
    """
    Automatically determine the versioned checkpoint directory for this run.
    Uses the config JSON filename as the base name and increments the version
    until an unused directory is found, so previous runs are never overwritten.

    Pattern: {CHECKPOINTS_DIR}/{model_config_name}_v{n}

    Example:
        checkpoints/xlm_roberta_base_v1  <- first run
        checkpoints/xlm_roberta_base_v2  <- config changed, re-run

    :param config_path: Path to the model config JSON file
    :return: Path to the next available versioned checkpoint directory
    """
    # Use the JSON filename (without extension) as the base name
    model_name = os.path.splitext(os.path.basename(config_path))[0]  # e.g. "xlm_roberta_base"

    version = 1
    while True:
        candidate = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_v{version}")
        if not os.path.exists(candidate):
            return candidate
        version += 1


if __name__ == "__main__":
    """
    End-to-end training pipeline. Steps in order:
        1. Load model config JSON and resolve versioned checkpoint directory
        2. Load parquet data, split into train/val/test and encode labels
        3. Build windowed datasets and DataLoaders
        4. Handle class imbalance via weighted loss and weighted sampler
        5. Initialize model dynamically from MODEL_REGISTRY
        6. Train, plot history, evaluate on test set
        7. Save test results, label mapping and config copy to checkpoint directory
    """
    model_config_path = "app/services/models/model_configs/xlm_roberta_base.json"
    data_path = "outputs/02-02-2026_17-38_5027_documents.parquet"

    # -- Load config ----------------------------------------------------------
    with open(model_config_path) as f:
        model_config = json.load(f)

    checkpoint_dir = resolve_checkpoint_dir(model_config_path)
    model_config["training"]["checkpoint_dir"] = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Config loaded : {model_config_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # -- Load data -------------------------------------------------------------
    data_path = data_path or model_config.get("data_path")
    if not data_path:
        raise ValueError("data_path'i train.py içinde tanımla veya config JSON'a 'data_path' ekle.")

    df = pd.read_parquet(data_path)
    print(f"\nLoaded {len(df)} documents from {data_path}")

    texts = df["text"].tolist()
    labels = df["document_type"].tolist()

    # -- Preprocess ------------------------------------------------------------
    preprocessor = DataPreprocessor(model_config["data"])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.prepare(texts, labels)

    num_classes = len(preprocessor.label_mapping)

    # Save label mapping alongside checkpoints
    os.makedirs(model_config["training"]["checkpoint_dir"], exist_ok=True)
    mapping_path = os.path.join(model_config["training"]["checkpoint_dir"], "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(preprocessor.label_mapping, f, ensure_ascii=False, indent=2)
    print(f"Label mapping saved → {mapping_path}")

    # -- Tokenizer & datasets --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_config["model"]["model_name"])

    train_dataset = WindowedDatasetCreator(X_train, y_train.tolist(), tokenizer,
                                           model_config["model"]["max_length"], model_config["data"]["stride"])
    val_dataset = WindowedDatasetCreator(X_val, y_val.tolist(), tokenizer,
                                         model_config["model"]["max_length"], model_config["data"]["stride"])
    test_dataset = WindowedDatasetCreator(X_test, y_test.tolist(), tokenizer,
                                          model_config["model"]["max_length"], model_config["data"]["stride"])

    # -- Class imbalance -------------------------------------------------------
    class_weights = None
    train_sampler = None

    if model_config["data"].get("use_class_weights"):
        class_weights = preprocessor.compute_class_weights(y_train)
        train_sampler = preprocessor.create_weighted_sampler(train_dataset.document_ids)

    batch = model_config["training"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                             num_workers=2, pin_memory=True)

    # -- Model -----------------------------------------------------------------
    ModelClass = load_model_class(model_config["model"]["type"])
    model = ModelClass(
        num_classes=num_classes,
        model_name=model_config["model"]["model_name"],
        dropout=model_config["model"].get("dropout", 0.1),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_config['model']['type']}  |  classes: {num_classes}  |  params: {total_params:,}")

    # -- Train -----------------------------------------------------------------
    trainer = Trainer(model, model_config, class_weights)
    trainer.train_loader = train_loader  # pass loader directly (used in _train_epoch)
    trainer.train(train_loader, val_loader)
    trainer.plot_history()

    # -- Test ------------------------------------------------------------------
    trainer.load_best()
    test_metrics, preds, true_labels = trainer.evaluate(
        test_loader,
        return_predictions=True,
        label_mapping=preprocessor.label_mapping,
    )

    print("\nTest results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test results
    results_path = os.path.join(model_config["training"]["checkpoint_dir"], "test_results.json")
    with open(results_path, "w") as f:
        json.dump({**test_metrics, "predictions": preds.tolist(), "labels": true_labels.tolist()}, f, indent=2)
    print(f"Test results saved → {results_path}")

    # Save final config copy
    config_path = os.path.join(model_config["training"]["checkpoint_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
