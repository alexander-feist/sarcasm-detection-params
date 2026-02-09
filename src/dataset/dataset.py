from collections.abc import Sequence

import numpy as np
import torch
from transformers import BatchEncoding, BertTokenizer


class Dataset(torch.utils.data.Dataset):
    """Dataset for training and evaluating the model."""

    def __init__(
        self,
        inputs: Sequence[str],
        labels: Sequence[int],
        tokenizer_name: str,
        max_tokens: int,
    ) -> None:
        """Initialize the dataset."""
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens

        self.labels = np.array(labels)
        self.inputs = [
            self.tokenizer(
                x,
                padding="max_length",
                max_length=self.max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            for x in inputs
        ]

    @property
    def classes(self) -> np.ndarray:
        """The class labels."""
        return self.labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[BatchEncoding, int]:
        """Return a sample from the dataset."""
        batch_x = self.inputs[idx]
        batch_y = self.labels[idx]
        return batch_x, batch_y
