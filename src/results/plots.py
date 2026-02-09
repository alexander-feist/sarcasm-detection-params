from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.results.utils import load_runs, total_training_time
from src.utils.utils import calculate_confidence_interval, path_from_root

MODEL_ORDER = ["tiny", "mini", "small", "medium", "base"]


def _model_sort_key(model: str) -> tuple[int, str]:
    name = model.lower()
    for idx, key in enumerate(MODEL_ORDER):
        if key in name:
            return idx, model
    return len(MODEL_ORDER), model


def _train_valid_df(runs: list[dict], metric: str) -> pd.DataFrame:
    """Build DataFrame and compute confidence intervals for train/valid curves."""
    rows = []
    for run in runs:
        model = run["model"]
        seed = run["seed"]
        for phase in ("train", "valid"):
            for epoch in run.get(phase, []):
                val = epoch.get(metric)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "phase": phase,
                        "epoch": int(epoch["epoch"]),
                        "value": float(val),
                    }
                )
    df = pd.DataFrame(rows)

    seed_level = df.groupby(["model", "seed", "phase", "epoch"], as_index=False)["value"].mean()

    stats = seed_level.groupby(["model", "phase", "epoch"])["value"].apply(
        lambda col: calculate_confidence_interval(col.to_numpy(), use_bootstrap=True)
    )

    return stats.apply(
        lambda t: pd.Series({"mean": t[0], "lower": t[2], "upper": t[3]})
    ).reset_index()


def plot_train_valid(
    runs: list[dict], metric: str, output_dir: Path, experiment_name: str
) -> None:
    """Plot train/valid curves with CI over seeds."""
    df = _train_valid_df(runs, metric)

    fig, ax = plt.subplots(figsize=(6, 4))

    models = sorted(df["model"].unique(), key=_model_sort_key)
    palette = sns.color_palette("tab10", n_colors=len(models))
    color_map = {m: palette[i] for i, m in enumerate(models)}

    for model in models:
        for phase in ("train", "valid"):
            sub = df[(df["model"] == model) & (df["phase"] == phase)]
            color = color_map[model]
            linestyle = "-" if phase == "train" else "--"
            label = model if phase == "train" else None
            ax.plot(sub["epoch"], sub["mean"], color=color, linestyle=linestyle, label=label)
            ax.fill_between(sub["epoch"], sub["lower"], sub["upper"], color=color, alpha=0.2)

    max_epoch = int(df["epoch"].max())
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(np.arange(0, max_epoch + 1, 1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    if len(experiment_name) > 0:
        fig.suptitle(experiment_name)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        (output_dir / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig((output_dir / ext / f"train_valid_{metric}").with_suffix(f".{ext}"))

    plt.close(fig)


def _test_seed_df(runs: list[dict], metric: str) -> pd.DataFrame:
    """Build DataFrame for test plots."""
    rows = []
    for run in runs:
        if metric == "training_time":
            value = total_training_time(run)
        elif metric == "training_time_include_valid":
            value = total_training_time(run, include_valid=True)
        elif metric == "training_time_best_epoch":
            value = total_training_time(run, best_epoch=run.get("metadata", {}).get("best_epoch"))
        elif metric == "best_epoch":
            value = run["metadata"].get("best_epoch")
        else:
            value = run["test"].get(metric)
        rows.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "value": float(value),
            }
        )
    return pd.DataFrame(rows)


def plot_test(runs: list[dict], metric: str, output_dir: Path, experiment_name: str) -> None:
    """Plot test results."""
    df_seed = _test_seed_df(runs, metric)

    fig, ax = plt.subplots(figsize=(6, 4))

    models = sorted(df_seed["model"].unique(), key=_model_sort_key)
    palette = sns.color_palette("tab10", n_colors=len(models))
    color_map = {m: palette[i] for i, m in enumerate(models)}

    sns.boxplot(
        x="model",
        y="value",
        data=df_seed,
        order=models,
        ax=ax,
        boxprops={"facecolor": "#f0f0f0", "edgecolor": "black"},
        medianprops={"linewidth": 2, "color": "black"},
        flierprops={"marker": "o", "markersize": 4, "color": "black"},
        width=0.4,
    )

    # Color medians per model
    for i, line in enumerate(ax.lines[4::6]):
        line.set_color(palette[i % len(palette)])

    sns.stripplot(
        x="model",
        y="value",
        data=df_seed,
        order=models,
        hue="model",
        palette=color_map,
        dodge=False,
        alpha=0.6,
        size=6,
        jitter=True,
        ax=ax,
        legend=False,
    )

    if metric in {"training_time", "training_time_include_valid"}:
        y_label = "Training Time (s)"
        file_name = f"total_{metric}"
    elif metric == "training_time_best_epoch":
        y_label = "Training Time Up To Best Epoch (s)"
        file_name = "total_training_time_best_epoch"
    elif metric == "best_epoch":
        y_label = "Best Epoch"
        file_name = "best_epoch"
        ax.set_ylim(0, 4.9)
        ax.set_yticks(np.arange(0, 5, 1))
    else:
        y_label = metric.replace("_", " ").title()
        file_name = f"test_{metric}"

    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.grid(visible=True, axis="y", alpha=0.3)

    if len(experiment_name) > 0:
        fig.suptitle(experiment_name)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        (output_dir / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig((output_dir / ext / file_name).with_suffix(f".{ext}"))

    plt.close(fig)


def create_plots(
    experiments_path: Path,
    output_dir: Path,
    train_metrics: list[str],
    test_metrics: list[str],
    experiment_name: str = "",
) -> None:
    """Create plots for all experiments under experiments_path and save them to output_dir."""
    runs = load_runs(experiments_path)

    if not runs:
        print("No runs found.")
        return

    for m in train_metrics:
        plot_train_valid(runs, m, output_dir, experiment_name)

    for m in test_metrics:
        plot_test(runs, m, output_dir, experiment_name)


if __name__ == "__main__":
    datasets = ["news_headlines", "isarcasm", "figlang"]
    for dataset in datasets:
        create_plots(
            experiments_path=path_from_root(f"results/logs/evaluation_{dataset}"),
            output_dir=path_from_root(f"results/plots/evaluation_{dataset}"),
            train_metrics=["loss", "accuracy", "f1", "precision", "recall"],
            test_metrics=[
                "accuracy",
                "f1",
                "precision",
                "recall",
                "training_time",
                "training_time_include_valid",
                "best_epoch",
                "training_time_best_epoch",
            ],
            experiment_name=dataset.replace("news_headlines", "News Headlines Dataset")
            .replace("isarcasm", "iSarcasm Dataset")
            .replace("figlang", "FigLang Dataset"),
        )
