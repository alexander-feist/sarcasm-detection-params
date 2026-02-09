import json
import uuid
from datetime import datetime
from typing import Any

from src.config import Config
from src.utils.device import get_device_info
from src.utils.utils import path_from_root


class Logger:
    """Logger for classification experiments."""

    def __init__(self, config: Config, output_dir_name: str) -> None:
        """Initialize the logger."""
        output_dir_name = output_dir_name.replace("/", "-")
        self.output_dir = path_from_root(f"results/logs/{output_dir_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = path_from_root(f"results/models/{output_dir_name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.id = uuid.uuid4().hex.upper()
        print(f"ID: {self.id}")

        self.data: dict[str, Any] = {
            "id": self.id,
            "device": get_device_info(),
            "config": config.to_dict(),
            "metadata": {},
            "results": {"train": [], "valid": [], "test": []},
        }
        self.filename = f"{self.id}.json"
        self.data["metadata"]["created_time"] = datetime.now().astimezone().isoformat()
        self._save_to_file()

    def _save_to_file(self) -> None:
        """Save the current data to the JSON file."""
        self.data["metadata"]["updated_time"] = datetime.now().astimezone().isoformat()
        filepath = self.output_dir / self.filename
        with filepath.open("w") as f:
            json.dump(self.data, f, indent=2)

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Log metadata and save the updated data to the JSON file."""
        self.data["metadata"].update(metadata)
        self._save_to_file()

    def log_train_result(self, result: dict[str, Any]) -> None:
        """Log the result of a train epoch and save the updated data to the JSON file."""
        self.data["results"]["train"].append(result)
        self._save_to_file()

    def log_valid_result(self, result: dict[str, Any]) -> None:
        """Log the result of a valid run and save the updated data to the JSON file."""
        self.data["results"]["valid"].append(result)
        self._save_to_file()

    def log_test_result(self, result: dict[str, Any]) -> None:
        """Log the result of a test run and save the updated data to the JSON file."""
        self.data["results"]["test"].append(result)
        self._save_to_file()
