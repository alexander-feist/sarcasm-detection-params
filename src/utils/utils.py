from pathlib import Path

import numpy as np
from scipy import stats
from seaborn.algorithms import bootstrap


def path_from_root(path: str) -> Path:
    """Get path from root directory."""
    return Path(__file__).parent.parent.parent / path


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95, *, use_bootstrap: bool = True
) -> tuple:
    """Calculate the confidence interval for the mean of the data."""
    mean = np.mean(data)

    if use_bootstrap:
        boot_means = bootstrap(data, n_boot=1000, seed=1)
        alpha = (1 - confidence) / 2
        lower_bound = np.percentile(boot_means, 100 * alpha)
        upper_bound = np.percentile(boot_means, 100 * (1 - alpha))
        margin = (upper_bound - lower_bound) / 2
    else:
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1.0 + confidence) / 2.0, len(data) - 1.0)
        lower_bound = mean - margin
        upper_bound = mean + margin

    return mean, margin, lower_bound, upper_bound
