import json
from pathlib import Path


def _normalize_models_by_config(runs: list[dict]) -> list[dict]:
    """Ensure that runs with the same "model" entry have the same config (except for the seed)."""
    model_config_map: dict[str, dict[str, str]] = {}

    for run in runs:
        base_model = run["model"]
        config = run.get("config", {})

        cfg_without_seed = config.copy()
        cfg_without_seed.pop("seed", None)

        signature = json.dumps(cfg_without_seed, sort_keys=True)

        model_map = model_config_map.setdefault(base_model, {})
        if signature not in model_map:
            new_name = base_model if not model_map else f"{base_model} ({len(model_map) + 1})"
            model_map[signature] = new_name

        run["model"] = model_map[signature]

    return runs


def load_runs(path: Path) -> list[dict]:
    """Load all metric JSON files under the given path into a list."""
    runs: list[dict] = []
    for filepath in sorted(path.rglob("*.json")):
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})
        results = data.get("results", {})

        model = str(config.get("pretrained_model_name"))
        seed = int(config.get("seed", 0))

        train = results.get("train", [])
        valid = results.get("valid", [])
        test = results.get("test", [])

        if len(test) == 0:
            print(f"No test results found for {data.get('id')}. Skipping.")
            continue
        test = test[-1]

        runs.append(
            {
                "model": model.replace("prajjwal1/", "").replace(
                    "google-bert/bert-base-uncased", "bert-base"
                ),
                "metadata": data.get("metadata", {}),
                "config": config,
                "seed": seed,
                "train": train,
                "valid": valid,
                "test": {k: float(v) for k, v in test.items() if k != "epoch"},
            }
        )
    return _normalize_models_by_config(runs)


def total_training_time(
    run: dict,
    *,
    include_valid: bool = False,
    best_epoch: int | None = None,
) -> float:
    """Sum train_time_taken.

    Optionally, include valid evaluate_time_taken.
    If best_epoch is provided, only sum up to that epoch (including it).

    """
    train_entries = run.get("train", [])
    valid_entries = run.get("valid", [])

    if best_epoch is not None:
        train_entries = train_entries[: best_epoch + 1]
        valid_entries = valid_entries[: best_epoch + 1]

    train = sum(float(e["train_time_taken"]) for e in train_entries)
    if not include_valid:
        return train

    valid = sum(float(e["evaluate_time_taken"]) for e in valid_entries)
    return train + valid
