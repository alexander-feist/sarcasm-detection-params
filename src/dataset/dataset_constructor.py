from typing import Protocol

from src.dataset.dataset import Dataset


class DatasetConstructor(Protocol):
    """Protocol for dataset constructors that take only a tokenizer name."""

    def __call__(self, tokenizer_name: str) -> Dataset:
        """Create a dataset instance from a tokenizer name."""
