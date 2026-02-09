import pandas as pd

from src.dataset.dataset import Dataset
from src.utils.utils import path_from_root

DATA_DIR_PATH = "src/dataset/data/isarcasm"


class ISarcasmDataset(Dataset):
    """iSarcasm dataset loader.

    References:
        https://github.com/AmirAbaskohi/SemEval2022-Task6-Sarcasm-Detection/blob/main/Data

    """

    def __init__(self, tokenizer_name: str) -> None:
        """Initialize the iSarcasm dataset."""
        data_dir = path_from_root(DATA_DIR_PATH)

        dfs = [pd.read_csv(path) for path in sorted(data_dir.glob("*.csv")) if path.is_file()]
        df = pd.concat(dfs, ignore_index=True)

        df["text"] = df["tweet"].astype(str)
        df["is_sarcastic"] = df["sarcastic"].astype(int)
        super().__init__(df["text"], df["is_sarcastic"], tokenizer_name, max_tokens=512)
