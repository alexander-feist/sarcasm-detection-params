from dataclasses import asdict, dataclass

from src.dataset.dataset_constructor import DatasetConstructor
from src.model.head_constructor import ClassificationHeadConstructor


@dataclass
class Config:
    """Configuration for training."""

    seed: int
    dataset: DatasetConstructor

    # Model
    pretrained_model_name: str
    classification_head: ClassificationHeadConstructor

    # Training
    train_test_split: float
    batch_size: int
    learning_rate: float
    max_grad_norm: float | None
    num_train_epochs: int

    def to_dict(self) -> dict:
        """Convert the config into a dict, using string fallbacks for non-primitive values."""
        raw = asdict(self)
        result = {}

        for key, value in raw.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
            elif key == "classification_head":
                result[key] = getattr(value, "spec", repr(value))
            else:
                result[key] = getattr(value, "__name__", repr(value))

        return result
