from pathlib import Path

import pandas as pd

from src.results.utils import load_runs, total_training_time
from src.utils.utils import path_from_root

MODEL_ORDER = ["tiny", "mini", "small", "medium", "base"]


def _escape_latex(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _model_sort_key(model: str) -> tuple[int, str]:
    name = model.lower()
    for idx, key in enumerate(MODEL_ORDER):
        if key in name:
            return idx, model
    return len(MODEL_ORDER), model


def _metric_value(run: dict, metric: str) -> float:
    if metric == "training_time":
        return float(total_training_time(run))
    if metric == "training_time_include_valid":
        return float(total_training_time(run, include_valid=True))
    if metric == "training_time_best_epoch":
        return float(
            total_training_time(run, best_epoch=run.get("metadata", {}).get("best_epoch"))
        )
    if metric == "best_epoch":
        return float(run.get("metadata", {}).get("best_epoch"))

    return float(run.get("test", {}).get(metric))


def _seed_df(experiments_path: Path, *, dataset: str, metrics: list[str]) -> pd.DataFrame:
    runs = load_runs(experiments_path)
    rows = [
        {
            "dataset": dataset,
            "model": str(run["model"]),
            "seed": int(run["seed"]),
            "metric": m,
            "value": _metric_value(run, m),
        }
        for run in runs
        for m in metrics
    ]
    return pd.DataFrame(rows)


def _stats_df(df_seed: pd.DataFrame) -> pd.DataFrame:
    per_seed = df_seed.groupby(["dataset", "model", "metric", "seed"], as_index=False)[
        "value"
    ].mean()

    group = per_seed.groupby(["dataset", "model", "metric"], sort=False)["value"]
    out = group.agg(mean="mean", std="std").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out.astype({"mean": float, "std": float})


def _metric_title(metric: str) -> str:
    if metric == "training_time":
        return "Training Time"
    if metric == "training_time_include_valid":
        return "Training Time (incl. Valid)"
    if metric == "training_time_best_epoch":
        return "Training Time (Best Epoch)"
    if metric == "best_epoch":
        return "Best Epoch"
    return metric.replace("_", " ").title()


def _siunitx_table_format(values: pd.Series) -> str:
    s = pd.Series(pd.to_numeric(values, errors="coerce")).dropna()
    max_abs = float(s.abs().max())
    int_digits = len(str(int(max_abs))) if max_abs >= 1 else 1
    dec_digits = 0 if (s % 1 == 0).all() else 3
    return f"{int_digits}.{dec_digits}"


def _colspec(metrics: list[str], *, df_stats: pd.DataFrame) -> str:
    parts = ["l", "X"]
    for m in metrics:
        m_stats = df_stats[df_stats["metric"] == m]

        if m == "best_epoch":
            parts.append(
                "S[table-format=2.0, group-digits=false]@{${}\\,\\pm\\,{}$}"
                "S[table-format=2.0, group-digits=false]"
            )
            continue

        tf_mean = _siunitx_table_format(m_stats["mean"])
        tf_std = _siunitx_table_format(m_stats["std"])
        parts.append(
            "S[table-format="
            + tf_mean
            + ", group-digits=false]@{${}\\,\\pm\\,{}$}S[table-format="
            + tf_std
            + ", group-digits=false]"
        )

    return " ".join(parts)


def _dataset_label(dataset: str, *, n_models: int) -> str:
    title = (
        dataset.replace("news_headlines", "News Headlines")
        .replace("isarcasm", "iSarcasm")
        .replace("figlang", "FigLang")
        .replace(" ", "\\\\")
    )
    return (
        rf"\multirow{{{n_models}}}{{*}}{{"
        rf"\parbox{{2.2cm}}{{\centering\rotatebox{{90}}{{\textbf{{\shortstack{{{title}}}}}}}}}"
        rf"}}"
    )


def generate_latex_metrics_table(datasets: list[str], metrics: list[str], out_path: str) -> None:
    """Generate a LaTeX table reporting the given metrics."""
    df_seed = pd.concat(
        [
            _seed_df(
                path_from_root(f"results/logs/evaluation_{ds}"),
                dataset=ds,
                metrics=metrics,
            )
            for ds in datasets
        ],
        ignore_index=True,
    )

    df_stats = _stats_df(df_seed)
    models = sorted(df_seed["model"].unique().tolist(), key=_model_sort_key)

    mean_tbl = df_stats.pivot_table(index=["dataset", "model"], columns="metric", values="mean")
    std_tbl = df_stats.pivot_table(index=["dataset", "model"], columns="metric", values="std")

    colspec = _colspec(metrics, df_stats=df_stats)

    header = (
        " & ".join(
            [
                r"\multicolumn{1}{l}{}",
                r"\multicolumn{1}{l}{\textbf{Model}}",
                *[
                    rf"\multicolumn{{2}}{{c}}{{\textbf{{{_escape_latex(_metric_title(m))}}}}}"
                    for m in metrics
                ],
            ]
        )
        + r" \\"
    )

    lines: list[str] = [
        "\\begin{tabularx}{\\linewidth}{" + colspec + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    n_models = len(models)

    for ds_i, ds in enumerate(datasets):
        ds_label = _dataset_label(ds, n_models=n_models)

        for i, model in enumerate(models):
            if (ds, model) not in mean_tbl.index:
                continue

            row: list[str] = [
                ds_label if i == 0 else "",
                rf"\texttt{{{_escape_latex(model)}}}",
            ]

            for m in metrics:
                mean = float(mean_tbl.loc[(ds, model), m])
                std = float(std_tbl.loc[(ds, model), m])

                mean_s = f"{round(mean)}" if m == "best_epoch" else f"{mean:.3f}"
                std_s = f"{round(std)}" if m == "best_epoch" else f"{std:.3f}"

                row += [mean_s, std_s]

            lines.append(" & ".join(row) + r" \\")

        if ds_i < len(datasets) - 1:
            lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
    ]

    out_file = path_from_root(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate_latex_metrics_table(
        datasets=["news_headlines", "isarcasm", "figlang"],
        metrics=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "training_time",
        ],
        out_path="results/tables/evaluation_metrics_table.tex",
    )
