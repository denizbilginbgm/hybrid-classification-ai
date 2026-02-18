from typing import List

import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer


class WindowedDatasetCreator(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: XLMRobertaTokenizer,
                 max_length: int = 512, stride: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Creating windows
        self.windows = []
        self.document_ids = []  # Which document does each window belong to?

        self._build_windows(texts, labels, stride)


    def _build_windows(self, texts: List[str], labels: List[int], stride: int = 256):
        """
        Tokenize every document and slice it into windows.
        Populates self.windows and self.document_ids.

        :param texts: Raw document texts
        :param labels: Encoded integer label for each document
        :param stride: Number of tokens to advance between consecutive windows
        """
        for doc_idx, (text, label) in enumerate(zip(texts, labels)):
            # Tokenize without truncation to get the full token sequence
            token_ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
            )["input_ids"]

            max_tokens = self.max_length - 2  # reserve slots for [CLS] and [SEP]

            if len(token_ids) <= max_tokens:
                # Fits in a single window
                self._add_window(token_ids, label, doc_idx)
            else:
                # Sliding window
                start = 0
                while start < len(token_ids):
                    end = min(start + max_tokens, len(token_ids))
                    self._add_window(token_ids[start:end], label, doc_idx)
                    if end == len(token_ids):
                        break
                    start += stride

    def _add_window(self, token_ids: list, label: int, doc_idx: int):
        """
        Wrap a token sequence with [CLS] and [SEP] and store it as a window.

        :param token_ids: Raw token IDs for this window, without special tokens
        :param label: Encoded integer label of the parent document
        :param doc_idx: Index of the parent document in the original texts list
        """
        window_ids = ([self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id])
        self.windows.append({
            "input_ids": window_ids,
            "label": label,
            "doc_id": doc_idx
        })
        self.document_ids.append(doc_idx)

    def __len__(self):
        """
        :return: Total number of windows across all documents (not number of documents)
        """
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Fetch a single window by index, pad it to max_length and return as tensors.

        :param idx: Window index
        :return: Dictionary with input_ids, attention_mask, labels and doc_id tensors
        """
        window = self.windows[idx]
        ids = window["input_ids"]

        # Pad to max_length
        pad_len = self.max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(window["label"], dtype=torch.long),
            "doc_id": window["doc_id"],
        }