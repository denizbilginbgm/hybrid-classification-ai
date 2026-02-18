from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTextClassifier(nn.Module, ABC):
    def __init__(self, num_classes: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Running centroids per class - updated during training, used for unknown detection
        self.register_buffer("class_centroids", torch.zeros(num_classes, hidden_size))
        self.register_buffer("centroid_counts", torch.zeros(num_classes))

    @abstractmethod
    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] (or equivalent) embeddings from the encoder.

        :param input_ids: Token IDs for each position in the sequence [batch_size, seq_len]
        :param attention_mask: Token IDs for each position in the sequence [batch_size, seq_len]
        :return: Embedding of the [CLS] token for each sample in the batch [batch_size, hidden_size]
        """
        pass

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_embeddings: bool = False):
        """
        Full forward pass: encoder → dropout → classifier.

        :param input_ids: Token IDs for each position in the sequence [batch_size, seq_len]
        :param attention_mask: Mask to avoid attending to padding tokens, 1 for real tokens 0 for padding [batch_size, seq_len]
        :param return_embeddings: If True, also returns raw [CLS] embeddings alongside logits (needed for centroid update & unknown detection)
        :return: logits over all classes [batch_size, num_classes], and optionally embeddings [batch_size, hidden_size] when return_embeddings=True
        """
        embeddings = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(self.dropout(embeddings))

        if return_embeddings:
            return logits, embeddings
        return logits

    @torch.no_grad()
    def update_centroids(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Update per-class centroids with a running average.
        Called every training batch so centroids stay current.
        Used later by UnknownDetector to flag out-of-distribution samples.

        :param embeddings: [CLS] embeddings for the current batch [batch_size, hidden_size]
        :param labels: Ground truth class indices for the current batch [batch_size]
        """
        for label in labels.unique():
            mask = labels == label
            batch_embs = embeddings[mask]

            count = self.centroid_counts[label]
            old_centroid = self.class_centroids[label]

            # Incremental mean: (old_sum + new_sum) / (old_count + new_count)
            new_centroid = (old_centroid * count + batch_embs.sum(0)) / (count + len(batch_embs))

            self.class_centroids[label] = new_centroid
            self.centroid_counts[label] += len(batch_embs)