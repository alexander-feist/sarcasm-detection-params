import random
from os import environ

import numpy as np
import torch


class RNG:
    """Random number generator manager."""

    _seed: int | None = None
    _np_generator: np.random.Generator | None = None

    @staticmethod
    def initialize(seed: int = 1) -> None:
        """Set global RNG seed for reproducibility."""
        RNG._seed = seed

        random.seed(seed)

        # NumPy
        RNG._np_generator = np.random.default_rng(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(mode=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_seed() -> int | None:
        """Get the RNG seed used for initialization."""
        return RNG._seed

    @staticmethod
    def np_generator() -> np.random.Generator:
        """Get the NumPy generator initialized with the RNG seed."""
        if RNG._np_generator is None:
            msg = "RNG is not initialized. Call RNG.initialize(...) first."
            raise RuntimeError(msg)
        return RNG._np_generator
