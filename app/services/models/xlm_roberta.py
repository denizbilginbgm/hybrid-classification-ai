import torch
from transformers import AutoModel, AutoConfig

from app.services.abstracts.base_text_classifier import BaseTextClassifier


class XlmRobertaTextClassifier(BaseTextClassifier):
    def __init__(self, num_classes: int, model_name: str, dropout: float = 0.1):
        hf_config = AutoConfig.from_pretrained(model_name)
        super().__init__(num_classes, hf_config.hidden_size, dropout)

        self.encoder = AutoModel.from_pretrained(model_name)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder and return the [CLS] token embedding as the document representation.
        The [CLS] token is always at position 0 and is designed to aggregate the full sequence meaning.

        :param input_ids: Token IDs for each position in the sequence [batch_size, seq_len]
        :param attention_mask: Mask to avoid attending to padding tokens, 1 for real tokens 0 for padding [batch_size, seq_len]
        :return: [CLS] token embedding for each sample in the batch [batch_size, hidden_size]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token