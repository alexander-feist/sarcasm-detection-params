import json

from src.utils.utils import path_from_root

LOGS_PATH = path_from_root("results/logs")
OUT_JSON = path_from_root("results/best_seeds_evaluation.json")

DATASET_CLASS_TO_ARG = {
    "NewsHeadlinesDataset": "news_headlines",
    "ISarcasmDataset": "isarcasm",
    "FigLangDataset": "figlang",
}


def main() -> None:
    """Create a JSON file with the best seeds from the evaluation.

    Evaluates all JSON logs and selects the run with the highest test F1 for each dataset-model
    pair. The output JSON contains the best seed from the evaluation per dataset-model pair.
    """
    best_f1_by_key = {}
    best_cfg_by_key = {}

    for logs_dir in LOGS_PATH.iterdir():
        if not logs_dir.is_dir() or not logs_dir.name.startswith("evaluation"):
            continue

        for log in sorted(logs_dir.glob("*.json")):
            obj = json.loads(log.read_text(encoding="utf-8"))
            f1 = float(obj["results"]["test"][0]["f1"])

            cfg = obj.get("config", {})
            model_name = cfg.get("pretrained_model_name")
            dataset_arg = DATASET_CLASS_TO_ARG.get(cfg.get("dataset"))

            model_short = model_name.split("/")[-1].replace("-", "_")
            key = f"{model_short}_{dataset_arg}"

            prev_best = best_f1_by_key.get(key)
            if (prev_best is None) or (f1 > prev_best):
                best_f1_by_key[key] = f1
                best_cfg_by_key[key] = {
                    "pretrained_model_name": model_name,
                    "dataset": dataset_arg,
                    "learning_rate": cfg.get("learning_rate"),
                    "batch_size": cfg.get("batch_size"),
                    "seed": cfg.get("seed"),
                    "best_test_f1": f1,
                    "log_file": log.name,
                }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(best_cfg_by_key, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
