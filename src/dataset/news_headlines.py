import pandas as pd

from src.dataset.dataset import Dataset
from src.utils.utils import path_from_root

DATA_DIR_PATH = "src/dataset/data/news-headlines"


class NewsHeadlinesDataset(Dataset):
    """News Headlines dataset.

    References:
        https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data

    """

    def __init__(self, tokenizer_name: str) -> None:
        """Initialize the News Headlines dataset."""
        data_dir = path_from_root(DATA_DIR_PATH)
        with (data_dir / "Sarcasm_Headlines_Dataset_v2.json").open("r", encoding="utf-8") as f:
            df = pd.read_json(f, lines=True)
            df["headline"] = df["headline"].astype(str)
            df = df.rename(columns={"headline": "text"})
            df["is_sarcastic"] = df["is_sarcastic"].astype(int)
            super().__init__(df.text, df.is_sarcastic, tokenizer_name, max_tokens=512)
