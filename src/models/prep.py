"""Train-only preprocessing for neural / linear scorers (trees take raw NaNs natively)."""
from __future__ import annotations

import numpy as np


class NanStandardizer:
    """Median-impute then standardise; statistics come ONLY from the array passed to fit()."""

    def fit(self, X: np.ndarray) -> "NanStandardizer":
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.mu_ = Xi.mean(0)
        sd = Xi.std(0)
        self.sd_ = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xi = np.where(np.isnan(X), self.med_, X)
        Z = (Xi - self.mu_) / self.sd_
        return np.clip(Z, -10, 10).astype(np.float32)  # clip: heavy-tailed BAF/Elliptic columns
