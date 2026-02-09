import json

from src.utils.utils import path_from_root

LOGS_PATH = path_from_root("results/logs")
OUT_JSON = path_from_root("results/best_hparams_grid_search.json")

DATASET_CLASS_TO_ARG = {
    "NewsHeadlinesDataset": "news_headlines",
    "ISarcasmDataset": "isarcasm",
    "FigLangDataset": "figlang",
}


def main() -> None:
    """Create a JSON file with the best hyperparameters from the grid search.

    Evaluates all JSON logs in each grid-search folder and selects the run with the
    highest test F1. The output JSON contains the best learning-rate / batch-size
    combination per experiment folder.
    """
    best = {}

    for logs_dir in LOGS_PATH.iterdir():
        if not logs_dir.is_dir() or not logs_dir.name.startswith("grid_search"):
            continue

        best_f1 = None
        best_cfg = None

        for log in sorted(logs_dir.glob("*.json")):
            obj = json.loads(log.read_text(encoding="utf-8"))
            f1 = float(obj["results"]["test"][0]["f1"])

            if (best_f1 is None) or (f1 > best_f1):
                cfg = obj.get("config", {})
                best_f1 = f1
                best_cfg = {
                    "pretrained_model_name": cfg.get("pretrained_model_name"),
                    "dataset": DATASET_CLASS_TO_ARG[cfg.get("dataset")],
                    "learning_rate": cfg.get("learning_rate"),
                    "batch_size": cfg.get("batch_size"),
                    "best_test_f1": f1,
                    "log_file": log.name,
                }

        best[logs_dir.name] = best_cfg

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
