import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.dataset.dataset import Dataset
from src.utils.utils import path_from_root

DATA_DIR_PATH = "src/dataset/data/figlang"


def _parse_jsonl(dir_path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    with dir_path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line.strip())

            label = obj.pop("label", None)
            if label == "SARCASM":
                obj["is_sarcastic"] = 1
            elif label == "NOT_SARCASM":
                obj["is_sarcastic"] = 0

            items.append(obj)

    return items


class FigLangDataset(Dataset):
    """FigLang dataset.

    References:
        https://github.com/EducationalTestingService/sarcasm

    """

    def __init__(
        self, tokenizer_name: str, source: Literal["reddit", "twitter", "both"] = "both"
    ) -> None:
        """Initialize the FigLang dataset."""
        data_dir = path_from_root(DATA_DIR_PATH)
        all_items: list[dict[str, Any]] = []

        pattern = "*.jsonl" if source == "both" else f"*{source}*.jsonl"
        for path in sorted(data_dir.glob(pattern)):
            if path.is_file():
                all_items.extend(_parse_jsonl(path))

        df = pd.DataFrame.from_records(all_items)
        df["text"] = df["response"].astype(str)
        df["is_sarcastic"] = df["is_sarcastic"].astype(int)

        super().__init__(df["text"], df["is_sarcastic"], tokenizer_name, max_tokens=512)
