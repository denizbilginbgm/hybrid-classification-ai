import importlib
import json
import os

import torch
from transformers import AutoTokenizer

from app.services.datasets.windowed_dataset_creator import WindowedDatasetCreator
from app.services.encoding.unknown_detector import UnknownDetector
from app.utils.utils import load_model_class

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)


def load_checkpoint(checkpoint_dir: str):
    """
    Load model config, label mapping and trained weights from a checkpoint directory.

    :param checkpoint_dir: Path to checkpoint directory (e.g. "checkpoints/xlm_roberta_base_v1")
    :return: Tuple of (model, config, label_mapping, device, unknown_detector_threshold)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Load label mapping
    mapping_path = os.path.join(checkpoint_dir, "label_mapping.json")
    with open(mapping_path) as f:
        label_mapping = json.load(f)
        # Convert string keys back to int
        label_mapping = {int(k): v for k, v in label_mapping.items()}

    # Initialize model
    num_classes = len(label_mapping)
    ModelClass = load_model_class(config["model"]["type"])
    model = ModelClass(
        num_classes=num_classes,
        model_name=config["model"]["model_name"],
        dropout=config["model"].get("dropout", 0.1),
    )

    # Load weights
    weights_path = os.path.join(checkpoint_dir, "best_model.pt")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"   Model loaded from {checkpoint_dir}")
    print(f"   Device: {device}")
    print(f"   Classes: {num_classes}")

    # Load unknown detector threshold if available
    unknown_detector_threshold = state.get("unknown_detector_threshold", None)

    return model, config, label_mapping, device, unknown_detector_threshold

def predict_single_text(text: str, model, config: dict, label_mapping: dict, device, unknown_detector=None) -> dict:
    """
    Predict the class of a single text document.

    :param text: Raw text to classify
    :param model: Trained model instance
    :param config: Model config dict
    :param label_mapping: Dict mapping encoded int labels to original class names
    :param device: torch device
    :param unknown_detector: Optional UnknownDetector instance for out-of-distribution detection
    :return: Dict with prediction results
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])

    # Create windowed dataset with a dummy label (won't be used)
    dataset = WindowedDatasetCreator(
        texts=[text],
        labels=[0],  # Dummy label
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        stride=config["data"]["stride"]
    )

    print(f"\n  Input text split into {len(dataset)} window(s)")

    # Collect logits from all windows
    all_logits = []
    all_embeddings = []

    with torch.no_grad():
        for window in dataset:
            input_ids = window["input_ids"].unsqueeze(0).to(device)  # [1, seq_len]
            attention_mask = window["attention_mask"].unsqueeze(0).to(device)

            logits, embeddings = model(input_ids, attention_mask, return_embeddings=True)

            all_logits.append(logits)
            all_embeddings.append(embeddings)

    # Aggregate windows using the same strategy as training
    all_logits = torch.cat(all_logits, dim=0)  # [num_windows, num_classes]
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [num_windows, hidden_size]

    pooling = config["data"]["window_pooling"]

    if pooling == "max":
        probs = torch.softmax(all_logits, dim=1)
        aggregated_logit = torch.log(torch.max(probs, dim=0).values + 1e-10)
        aggregated_embedding = all_embeddings.mean(dim=0)  # Average embeddings for unknown detection
    elif pooling == "mean":
        aggregated_logit = all_logits.mean(dim=0)
        aggregated_embedding = all_embeddings.mean(dim=0)
    elif pooling == "first":
        aggregated_logit = all_logits[0]
        aggregated_embedding = all_embeddings[0]
    else:
        raise ValueError(f"Unknown window_pooling: {pooling}")

    # Get prediction
    aggregated_logit = aggregated_logit.unsqueeze(0)  # [1, num_classes]
    aggregated_embedding = aggregated_embedding.unsqueeze(0)  # [1, hidden_size]

    probs = torch.softmax(aggregated_logit, dim=1)
    predicted_class = torch.argmax(aggregated_logit, dim=1).item()
    confidence = probs[0, predicted_class].item()

    # Check if unknown
    is_unknown = False
    if unknown_detector is not None:
        unknown_mask = unknown_detector.predict(aggregated_logit, aggregated_embedding, model)
        is_unknown = unknown_mask[0].item()

    return {
        "predicted_class_id": predicted_class,
        "predicted_class_name": label_mapping[predicted_class],
        "confidence": confidence,
        "is_unknown": is_unknown,
        "all_probabilities": {
            label_mapping[i]: probs[0, i].item()
            for i in range(len(label_mapping))
        }
    }

if __name__ == "__main__":
    checkpoint_dir = "checkpoints/xlm_roberta_base_v1"
    text = """
    Bu bir örnek fatura metnidir. 
    Toplam tutar: 1500 TL
    KDV dahil fiyat hesaplanmıştır.
    Ödeme vadesi: 30 gün
    """

    # Load model
    model, config, label_mapping, device, unknown_detector_threshold = load_checkpoint(checkpoint_dir)

    # Initialize unknown detector if enabled in config
    unknown_detector = None
    if config["unknown_detection"].get("enabled"):
        unknown_detector = UnknownDetector(config["unknown_detection"])

        if unknown_detector_threshold is not None:
            unknown_detector.distance_threshold = unknown_detector_threshold
            print(f"Unknown detection enabled (threshold: {unknown_detector_threshold:.4f})")
        else:
            print("Unknown detection is enabled but threshold was not saved in checkpoint.")
            print("Model was likely trained before this feature was added. Unknown detection disabled.")
            unknown_detector = None

    # Predict
    result = predict_single_text(text, model, config, label_mapping, device, unknown_detector)

    # Display results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Predicted class: {result['predicted_class_name']}")
    print(f"Confidence:      {result['confidence']:.2%}")
    if result['is_unknown']:
        print(f"UNKNOWN, This document may not belong to any known class")

    print("\nAll probabilities:")
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for class_name, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        print(f"  {class_name:30s} {prob:6.2%} {bar}")