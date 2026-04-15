# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DTModel: Abstract base class for all Digital Twin model implementations.

Provides a uniform interface for training, prediction, and serialization
across all RF Digital Twin algorithms (Bayesian ExactGP, SVGP, LightGBM,
Neural Beam Field, GNN, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DTModelType(Enum):
    """Registry of supported Digital Twin model types."""

    BAYESIAN = "bayesian"
    SVGP = "svgp"
    # Future: LIGHTGBM = "lightgbm"
    # Future: NEURAL_BEAM_FIELD = "neural_beam_field"
    # Future: GNN = "gnn"


@dataclass
class TrainConfig:
    """Base training configuration. Subclass for model-specific parameters.

    Example subclass::

        @dataclass
        class SVGPTrainConfig(TrainConfig):
            num_inducing_points: int = 500
            batch_size: int = 256
    """

    max_iter: int = 100
    learning_rate: float = 0.05
    stopping_threshold: float = 1e-4
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictConfig:
    """Base prediction configuration. Subclass for model-specific parameters."""

    extra: Dict[str, Any] = field(default_factory=dict)


class DTModel(ABC):
    """Abstract base class for all Digital Twin model implementations.

    Each instance represents one trained model (typically per-cell).
    The caller is responsible for managing the ``Dict[str, DTModel]``
    cell-to-model mapping.

    Lifecycle::

        model = ConcreteModel(...)
        loss = model.train(data_in, x_columns, y_columns, config)
        means, uncertainties = model.predict(prediction_dfs)
        model.save("/path/to/model.pt")

        model2 = ConcreteModel(...)
        model2.load("/path/to/model.pt")
        means2, _ = model2.predict(prediction_dfs)
    """

    @property
    @abstractmethod
    def model_type(self) -> DTModelType:
        """Return the DTModelType enum value for this model."""
        ...

    @abstractmethod
    def train(
        self,
        data_in: List[pd.DataFrame],
        x_columns: List[str],
        y_columns: List[str],
        config: Optional[TrainConfig] = None,
        **kwargs,
    ) -> np.ndarray:
        """Train the model on the provided data.

        Args:
            data_in: List of DataFrames, one per cell. Each DataFrame
                contains both feature columns and target columns.
            x_columns: Feature column names to use for training.
            y_columns: Target column names (typically ``["avg_rsrp"]``).
            config: Training hyperparameters. Uses model-specific defaults
                if None.
            **kwargs: Extension point for future models (e.g., GNN may
                pass ``topology_graph=...``).

        Returns:
            Loss array over training iterations as a numpy ndarray.
        """
        ...

    @abstractmethod
    def predict(
        self,
        prediction_dfs: List[pd.DataFrame],
        config: Optional[PredictConfig] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for the provided data.

        Side effect: implementations may mutate ``prediction_dfs`` in-place
        by adding ``RXPOWER_DBM`` and ``RXPOWER_STDDEV_DBM`` columns
        (consistent with BayesianDigitalTwin behavior).

        Args:
            prediction_dfs: List of DataFrames, one per cell. Must contain
                the same feature columns used during training.
            config: Prediction configuration. Uses model-specific defaults
                if None.
            **kwargs: Extension point for future models.

        Returns:
            Tuple of (means, uncertainties), each a numpy ndarray of shape
            ``[num_cells, num_locations]``.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize the trained model to disk.

        Args:
            path: File path to write the model to.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: File path to read the model from.
        """
        ...

    @property
    def is_trained(self) -> bool:
        """True if the model has been successfully trained."""
        return getattr(self, "_trained", False)
