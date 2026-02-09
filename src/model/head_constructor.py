from typing import Protocol

from src.model.classification_heads import ClassificationHead


class ClassificationHeadConstructor(Protocol):
    """Protocol for classification head constructors."""

    def __call__(self, hidden_size: int) -> ClassificationHead:
        """Create a classification head instance with the given hidden size."""
