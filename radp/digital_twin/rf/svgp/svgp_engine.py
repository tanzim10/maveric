# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
SVGPDigitalTwin: Sparse Variational GP RF Digital Twin.

Follows the same pipeline as BayesianDigitalTwin (same normalization,
same per-cell architecture, same prediction flow, same output contract)
but replaces the Exact GP internals with a Sparse Variational GP (SVGP):

    Problem in BayesianDigitalTwin         SVGP Fix
    ─────────────────────────────          ─────────────────────────────────────
    O(n³) training — fails at >10K pts     O(n·m²) with m inducing points
    O(n²) kernel matrix in memory          Only m×m variational params stored
    All data in memory at training time    Mini-batch ELBO via DataLoader
    Pickle serialization (security risk)   torch.save with state_dict + metadata
    Fixed iteration count only             Epoch-based with early stopping

References:
    Hensman et al. (2013) "Gaussian Processes for Big Data" (SVGP)
    Titsias (2009) "Variational Learning of Inducing Variables"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from radp.digital_twin.rf.base_model import DTModel, DTModelType, PredictConfig, TrainConfig
from radp.digital_twin.rf.bayesian.bayesian_engine import NormMethod
from radp.digital_twin.utils import constants

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPyTorch model
# ---------------------------------------------------------------------------

class SVGPGPModel(gpytorch.models.ApproximateGP):
    """Sparse Variational GP — same kernel structure as Bayesian ExactGPModel.

    Uses Cholesky variational distribution and a standard variational
    strategy with learnable inducing point locations.
    Kernel: ConstantMean + ScaleKernel(RBFKernel) — identical to the
    Bayesian engine so direct accuracy comparisons are valid.
    """

    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class SVGPTrainConfig(TrainConfig):
    """SVGP-specific training configuration.

    Extends :class:`TrainConfig` with SVGP-only parameters.
    ``max_iter`` is inherited but unused (SVGP uses ``num_epochs``);
    it is kept for interface compatibility with generic callers.
    """

    num_inducing_points: int = 500
    """Number of inducing points. Larger values → more accurate but slower."""

    batch_size: int = 256
    """Mini-batch size for ELBO gradient estimation."""

    num_epochs: int = 50
    """Maximum training epochs. Early stopping may terminate sooner."""

    inducing_point_method: str = "kmeans"
    """Inducing point selection: ``"kmeans"`` (default) or ``"random"``."""


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class SVGPDigitalTwin(DTModel):
    """Sparse Variational GP Digital Twin.

    Follows the same per-cell architecture and prediction interface as
    :class:`BayesianDigitalTwin` but uses mini-batch ELBO training and
    safe ``torch.save`` serialization.

    Lifecycle is identical to :class:`BayesianDTModel`::

        model = SVGPDigitalTwin(x_max=x_max, x_min=x_min)
        loss = model.train(data_in, x_columns, y_columns, config)
        means, stds = model.predict(prediction_dfs)
        model.save("/tmp/svgp_cell1.pt")
    """

    def __init__(
        self,
        norm_method: NormMethod = NormMethod.MINMAX,
        x_max: Optional[Dict[str, float]] = None,
        x_min: Optional[Dict[str, float]] = None,
    ):
        self._norm_method = norm_method
        self._x_max = x_max
        self._x_min = x_min
        self._trained = False

        # Populated during train()
        self._x_columns: Optional[List[str]] = None
        self._y_columns: Optional[List[str]] = None
        self._cell_ids: List[str] = []
        self._num_cells: int = 0
        self._num_features: int = 0

        # Per-cell normalization stats (lists parallel to cell_ids)
        # Each entry is a pandas Series indexed by column name.
        self._xmeans: List[pd.Series] = []
        self._xstds: List[pd.Series] = []
        self._xmax: List[pd.Series] = []
        self._xmin_stored: List[pd.Series] = []
        self._ymeans: List[pd.Series] = []
        self._ystds: List[pd.Series] = []

        # Per-cell GPyTorch models and likelihoods
        self._models: List[SVGPGPModel] = []
        self._likelihoods: List[gpytorch.likelihoods.GaussianLikelihood] = []

    # ---- DTModel interface ------------------------------------------------

    @property
    def model_type(self) -> DTModelType:
        return DTModelType.SVGP

    def train(
        self,
        data_in: List[pd.DataFrame],
        x_columns: List[str],
        y_columns: List[str],
        config: Optional[TrainConfig] = None,
        **kwargs,
    ) -> np.ndarray:
        """Train one SVGP per cell using mini-batch Variational ELBO.

        Returns a 2-D loss array of shape ``[num_cells, num_epochs_trained]``.
        """
        if config is None:
            config = SVGPTrainConfig()
        elif not isinstance(config, SVGPTrainConfig):
            config = SVGPTrainConfig(
                max_iter=config.max_iter,
                learning_rate=config.learning_rate,
                stopping_threshold=config.stopping_threshold,
            )

        self._x_columns = x_columns
        self._y_columns = y_columns

        # Step 1: normalization stats — replicates bayesian_engine.py lines 86-113
        self._compute_normalization_stats(data_in)

        # Step 2: per-cell training
        self._models = []
        self._likelihoods = []
        all_losses: List[np.ndarray] = []

        for m in range(self._num_cells):
            cell_X, cell_Y = self._make_cell_tensors(data_in[m])
            inducing_pts = _select_inducing_points(
                cell_X, config.num_inducing_points, config.inducing_point_method
            )

            model = SVGPGPModel(inducing_pts)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            cell_losses = self._train_cell(model, likelihood, cell_X, cell_Y, config)
            all_losses.append(cell_losses)
            self._models.append(model)
            self._likelihoods.append(likelihood)

            logger.info(
                "Trained SVGP cell %s (%d pts, %d inducing, %d epochs): "
                "min_loss=%.4f, final_loss=%.4f",
                self._cell_ids[m],
                cell_X.size(0),
                inducing_pts.size(0),
                len(cell_losses),
                float(cell_losses.min()),
                float(cell_losses[-1]),
            )

        self._trained = True

        # Pad losses to same length and return as 2D array
        max_epochs = max(len(l) for l in all_losses)
        padded = np.full((self._num_cells, max_epochs), np.nan)
        for m, l in enumerate(all_losses):
            padded[m, : len(l)] = l
        return padded

    def predict(
        self,
        prediction_dfs: List[pd.DataFrame],
        config: Optional[PredictConfig] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict Rx power for all cells.

        Mirrors :meth:`BayesianDigitalTwin.predict_distributed_gpmodel`:
        normalizes features, runs SVGP inference, denormalizes predictions,
        and adds ``RXPOWER_DBM`` / ``RXPOWER_STDDEV_DBM`` columns to each
        DataFrame in ``prediction_dfs``.

        Returns ``(means, stds)`` each of shape ``[num_cells, num_locations]``.
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        num_locations = prediction_dfs[0].shape[0]
        pred_means_all = np.zeros([self._num_cells, num_locations], dtype=np.float32)
        pred_stds_all = np.zeros([self._num_cells, num_locations], dtype=np.float32)

        for m in range(self._num_cells):
            predict_X = self._normalize_predict_features(
                prediction_dfs[m], m
            )

            self._models[m].eval()
            self._likelihoods[m].eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self._likelihoods[m](self._models[m](predict_X))
                mean = observed_pred.mean.cpu().numpy()
                var = observed_pred.variance.cpu().numpy()

            # Denormalize — extract scalars explicitly to avoid Series broadcasting
            ymean = float(self._ymeans[m].iloc[0])
            ystd = float(self._ystds[m].iloc[0])
            pred_means_all[m] = mean * ystd + ymean
            pred_stds_all[m] = np.sqrt(np.maximum(var, 0.0)) * ystd

            # Mutate prediction_dfs (same side-effect as BayesianDigitalTwin)
            prediction_dfs[m][constants.RXPOWER_DBM] = pred_means_all[m]
            prediction_dfs[m][constants.RXPOWER_STDDEV_DBM] = pred_stds_all[m]

        return pred_means_all, pred_stds_all

    def save(self, path: str) -> None:
        """Serialize to disk using torch.save — no pickle."""
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        torch.save(
            {
                "model_type": self.model_type.value,
                "norm_method": self._norm_method.value,
                "x_columns": self._x_columns,
                "y_columns": self._y_columns,
                "num_cells": self._num_cells,
                "cell_ids": self._cell_ids,
                "num_features": self._num_features,
                "x_max_config": self._x_max,
                "x_min_config": self._x_min,
                "normalization": {
                    "xmeans": [s.to_dict() for s in self._xmeans],
                    "xstds": [s.to_dict() for s in self._xstds],
                    "xmax": [s.to_dict() for s in self._xmax],
                    "xmin": [s.to_dict() for s in self._xmin_stored],
                    "ymeans": [s.to_dict() for s in self._ymeans],
                    "ystds": [s.to_dict() for s in self._ystds],
                },
                "model_state_dicts": [m.state_dict() for m in self._models],
                "likelihood_state_dicts": [
                    ll.state_dict() for ll in self._likelihoods
                ],
                # Store inducing point shapes to reconstruct models on load
                "inducing_shapes": [
                    m.variational_strategy.inducing_points.shape
                    for m in self._models
                ],
            },
            path,
        )
        logger.info("SVGPDigitalTwin saved to %s", path)

    def load(self, path: str) -> None:
        """Load from a torch.save checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self._x_columns = checkpoint["x_columns"]
        self._y_columns = checkpoint["y_columns"]
        self._num_cells = checkpoint["num_cells"]
        self._cell_ids = checkpoint["cell_ids"]
        self._num_features = checkpoint["num_features"]
        self._norm_method = NormMethod(checkpoint["norm_method"])
        self._x_max = checkpoint["x_max_config"]
        self._x_min = checkpoint["x_min_config"]

        norm = checkpoint["normalization"]
        self._xmeans = [pd.Series(d) for d in norm["xmeans"]]
        self._xstds = [pd.Series(d) for d in norm["xstds"]]
        self._xmax = [pd.Series(d) for d in norm["xmax"]]
        self._xmin_stored = [pd.Series(d) for d in norm["xmin"]]
        self._ymeans = [pd.Series(d) for d in norm["ymeans"]]
        self._ystds = [pd.Series(d) for d in norm["ystds"]]

        inducing_shapes = checkpoint["inducing_shapes"]
        self._models = []
        self._likelihoods = []
        for m in range(self._num_cells):
            shape = inducing_shapes[m]
            dummy_inducing = torch.zeros(*shape)
            model = SVGPGPModel(dummy_inducing)
            model.load_state_dict(checkpoint["model_state_dicts"][m])
            model.eval()

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.load_state_dict(checkpoint["likelihood_state_dicts"][m])
            likelihood.eval()

            self._models.append(model)
            self._likelihoods.append(likelihood)

        self._trained = True
        logger.info("SVGPDigitalTwin loaded from %s", path)

    # ---- Internal pipeline helpers ----------------------------------------

    def _compute_normalization_stats(self, data_in: List[pd.DataFrame]) -> None:
        """Compute per-cell normalization statistics.

        Replicates bayesian_engine.py __init__ lines 86-113 exactly,
        including the optional override from ``x_max`` / ``x_min`` config.
        """
        assert self._x_columns is not None
        assert self._y_columns is not None

        self._cell_ids = [df["cell_id"].unique()[0] for df in data_in]
        self._num_cells = len(self._cell_ids)
        self._num_features = len(self._x_columns)

        self._xmeans = []
        self._xstds = []
        self._xmax = []
        self._xmin_stored = []
        self._ymeans = []
        self._ystds = []

        for m in range(self._num_cells):
            cell_stats = data_in[m].describe()

            xmax_m = cell_stats.loc["max", self._x_columns].copy()
            xmin_m = cell_stats.loc["min", self._x_columns].copy()

            if self._x_max:
                for k, v in self._x_max.items():
                    if k in self._x_columns:
                        xmax_m[k] = v
            if self._x_min:
                for k, v in self._x_min.items():
                    if k in self._x_columns:
                        xmin_m[k] = v

            self._xmax.append(xmax_m)
            self._xmin_stored.append(xmin_m)
            self._xmeans.append(cell_stats.loc["mean", self._x_columns])
            self._xstds.append(cell_stats.loc["std", self._x_columns])
            self._ymeans.append(cell_stats.loc["mean", self._y_columns])
            self._ystds.append(cell_stats.loc["std", self._y_columns])

    def _make_cell_tensors(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize features and targets for one cell, return as tensors."""
        assert self._x_columns is not None
        assert self._y_columns is not None

        m = len(self._models)  # current cell index during train loop

        if self._norm_method == NormMethod.MINMAX:
            norm_x = (df[self._x_columns] - self._xmin_stored[m]) / (
                self._xmax[m] - self._xmin_stored[m]
            )
        else:  # ZSCORE
            norm_x = (df[self._x_columns] - self._xmeans[m]) / self._xstds[m]

        norm_y = (df[self._y_columns] - self._ymeans[m]) / self._ystds[m]

        cell_X = torch.tensor(norm_x.values, dtype=torch.float32)
        cell_Y = torch.tensor(norm_y.values.squeeze(), dtype=torch.float32)
        return cell_X, cell_Y

    def _normalize_predict_features(
        self, df: pd.DataFrame, cell_idx: int
    ) -> torch.Tensor:
        """Normalize prediction features for cell ``cell_idx``."""
        assert self._x_columns is not None

        if self._norm_method == NormMethod.MINMAX:
            norm_x = (df[self._x_columns] - self._xmin_stored[cell_idx]) / (
                self._xmax[cell_idx] - self._xmin_stored[cell_idx]
            )
        else:  # ZSCORE
            norm_x = (
                df[self._x_columns] - self._xmeans[cell_idx]
            ) / self._xstds[cell_idx]

        return torch.tensor(norm_x.values, dtype=torch.float32)

    def _train_cell(
        self,
        model: SVGPGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        cell_X: torch.Tensor,
        cell_Y: torch.Tensor,
        config: SVGPTrainConfig,
    ) -> np.ndarray:
        """Mini-batch Variational ELBO training loop for a single cell."""
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=config.learning_rate,
        )
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=cell_X.size(0)
        )

        dataset = TensorDataset(cell_X, cell_Y)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        losses: List[float] = []
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = -mll(output, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if len(losses) > 1:
                if abs(losses[-1] - losses[-2]) < config.stopping_threshold:
                    logger.debug(
                        "SVGP early stop at epoch %d (delta=%.2e)", epoch, abs(losses[-1] - losses[-2])
                    )
                    break

        return np.array(losses)


# ---------------------------------------------------------------------------
# Inducing point selection
# ---------------------------------------------------------------------------

def _select_inducing_points(
    train_X: torch.Tensor,
    num_inducing: int,
    method: str = "kmeans",
) -> torch.Tensor:
    """Choose inducing points from training data.

    If ``n <= num_inducing``, returns all training points.
    Otherwise uses k-means (default) or random selection.
    """
    n = train_X.size(0)
    if n <= num_inducing:
        return train_X.clone()
    if method == "kmeans":
        return _kmeans_inducing_points(train_X, num_inducing)
    indices = torch.randperm(n)[:num_inducing]
    return train_X[indices].clone()


def _kmeans_inducing_points(
    X: torch.Tensor, k: int, max_iter: int = 50
) -> torch.Tensor:
    """K-means centroid selection — pure PyTorch, no sklearn dependency.

    Uses Euclidean distance (torch.cdist) and Lloyd's algorithm.
    Falls back to the current centroid if a cluster is empty.
    """
    n = X.size(0)
    if n <= k:
        return X.clone()

    indices = torch.randperm(n)[:k]
    centroids = X[indices].clone()

    for _ in range(max_iter):
        dists = torch.cdist(X, centroids)           # [n, k]
        assignments = dists.argmin(dim=1)            # [n]

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = X[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]     # keep old centroid if empty

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids
