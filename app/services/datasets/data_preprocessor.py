from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_mapping: dict = {} # encoded int to original label name

    def prepare(self, texts: List[str], labels: List) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Encode labels, split data into train/val/test, and optionally oversample minority classes.
        Oversampling is applied to the training set only to prevent data leakage into val/test.

        :param texts: Raw document texts
        :param labels: Original class labels, can be any type (str, int, etc.)
        :return: Three tuples of (texts, labels) for train, val and test splits respectively
        """
        # Encode
        encoded = self.label_encoder.fit_transform(labels)
        self.label_mapping = {
            int(enc): str(orig)
            for enc, orig in enumerate(self.label_encoder.classes_)
        }

        self._print_distribution(encoded)

        # Split
        test_size = self.config["test_size"]
        val_size = self.config["val_size"]

        X_tv, X_test, y_tv, y_test = train_test_split(
            texts, encoded,
            test_size=test_size,
            random_state=self.config["random_state"],
            stratify=encoded,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_size / (1 - test_size),
            random_state=self.config["random_state"],
            stratify=y_tv,
        )

        # Oversample
        if self.config.get("use_minority_oversampling", False):
            X_train, y_train = self._oversample(X_train, y_train.tolist())
            y_train = np.array(y_train)

        print(f"\nSplit — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        Compute balanced class weights to pass to the loss function.
        Rare classes get higher weights so the model is penalized more for misclassifying them.

        :param labels: Encoded integer labels of the training set [n_samples]
        :return: Weight tensor with one value per class [n_classes], placed on the correct device
        """
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)

        print("\nClass weights:")
        for cls, w in enumerate(weights):
            print(f"  {self.label_mapping.get(cls, cls):>30s} → {w:.3f}  (n={np.sum(labels == cls)})")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(weights, dtype=torch.float, device=device)

    def create_weighted_sampler(self, doc_labels: List[int]) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler so every class appears roughly equally in each batch.
        Each sample's weight is the inverse of its class frequency: 1 / class_count.
        Works on document-level labels, not window-level, to avoid over-representing
        documents that were split into many windows.

        :param doc_labels: One encoded integer label per document in the training set
        :return: WeightedRandomSampler instance to pass directly to DataLoader
        """
        counts = Counter(doc_labels)
        weights = [1.0 / counts[lbl] for lbl in doc_labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def _oversample(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """
        Duplicate samples from minority classes until they reach the threshold.
        Classes with fewer samples than minority_threshold are duplicated oversample_factor times.

        :param texts: Training texts to oversample
        :param labels: Encoded integer labels corresponding to texts
        :return: Augmented texts and labels with minority classes duplicated
        """
        threshold = self.config["minority_threshold"]
        factor = self.config["oversample_factor"]
        counts = Counter(labels)

        texts_out, labels_out = list(texts), list(labels)

        for lbl, count in counts.items():
            if count < threshold:
                indices = [i for i, l in enumerate(labels) if l == lbl]
                for _ in range(factor - 1):
                    for i in indices:
                        texts_out.append(texts[i])
                        labels_out.append(labels[i])
                print(f"  Oversampled class {self.label_mapping.get(lbl, lbl)}: {count} → {count * factor}")

        return texts_out, labels_out

    def _print_distribution(self, labels: np.ndarray):
        """
        Print the number of samples per class to the console.

        :param labels: Encoded integer labels [n_samples]
        """
        print("\nLabel distribution:")
        for enc, orig in sorted(self.label_mapping.items()):
            print(f"  [{enc}] {orig:>30s}: {np.sum(labels == enc)}")