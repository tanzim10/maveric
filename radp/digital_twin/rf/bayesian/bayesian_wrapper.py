# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BayesianDTModel: DTModel wrapper for BayesianDigitalTwin.

This is a composition-based adapter. It does NOT modify bayesian_engine.py
in any way. It creates a BayesianDigitalTwin instance internally and
delegates all calls to it, adapting the interface to the DTModel ABC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from radp.digital_twin.rf.base_model import DTModel, DTModelType, PredictConfig, TrainConfig
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod

logger = logging.getLogger(__name__)


@dataclass
class BayesianTrainConfig(TrainConfig):
    """Bayesian-specific training config.

    Maps to :meth:`BayesianDigitalTwin.train_distributed_gpmodel` parameters.
    The base class fields ``max_iter``, ``learning_rate``, and
    ``stopping_threshold`` map to ``maxiter``, ``lr``, and
    ``stopping_threshold`` respectively.
    """

    load_model: bool = False
    save_model: bool = False
    model_path: Optional[str] = None
    model_name: Optional[str] = None


class BayesianDTModel(DTModel):
    """Adapter that wraps BayesianDigitalTwin to implement the DTModel interface.

    Does NOT subclass or modify :class:`BayesianDigitalTwin`. Holds an
    instance internally and delegates all method calls to it.

    Usage::

        wrapper = BayesianDTModel(x_max=x_max, x_min=x_min)
        loss = wrapper.train(data_in, x_columns, y_columns)
        means, stds = wrapper.predict(prediction_dfs)

        # Access the raw engine when Bayesian-specific methods are needed
        wrapper.engine.preprocess_ue_training_data(...)
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
        self._engine: Optional[BayesianDigitalTwin] = None
        self._training_data: Optional[List[pd.DataFrame]] = None
        self._trained = False

    # ---- DTModel interface ------------------------------------------------

    @property
    def model_type(self) -> DTModelType:
        return DTModelType.BAYESIAN

    def train(
        self,
        data_in: List[pd.DataFrame],
        x_columns: List[str],
        y_columns: List[str],
        config: Optional[TrainConfig] = None,
        **kwargs,
    ) -> np.ndarray:
        """Create a BayesianDigitalTwin and train it.

        The engine is created lazily here (not in ``__init__``) because
        BayesianDigitalTwin requires training data at construction time.
        """
        if config is None:
            config = BayesianTrainConfig()

        # Cache training data — ExactGP bakes it into the prediction strategy,
        # so it must be available for save/load roundtrips.
        self._training_data = data_in

        self._engine = BayesianDigitalTwin(
            data_in=data_in,
            x_columns=x_columns,
            y_columns=y_columns,
            norm_method=self._norm_method,
            x_max=self._x_max,
            x_min=self._x_min,
        )

        train_kwargs: Dict = dict(
            maxiter=config.max_iter,
            lr=config.learning_rate,
            stopping_threshold=config.stopping_threshold,
        )
        if isinstance(config, BayesianTrainConfig):
            train_kwargs.update(
                load_model=config.load_model,
                save_model=config.save_model,
                model_path=config.model_path,
                model_name=config.model_name,
            )

        loss = self._engine.train_distributed_gpmodel(**train_kwargs)
        self._trained = True
        return loss

    def predict(
        self,
        prediction_dfs: List[pd.DataFrame],
        config: Optional[PredictConfig] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate prediction to BayesianDigitalTwin.

        Also mutates ``prediction_dfs`` in-place (adds ``RXPOWER_DBM`` and
        ``RXPOWER_STDDEV_DBM`` columns), consistent with the engine's own
        behavior.
        """
        return self.engine.predict_distributed_gpmodel(prediction_dfs)

    def save(self, path: str) -> None:
        """Serialize model to disk using torch.save (no pickle).

        ExactGP bakes training data into its prediction strategy, so we
        include the raw training DataFrames as records. This keeps Bayesian
        save/load correct without re-introducing pickle.
        """
        engine = self.engine
        if self._training_data is None:
            raise RuntimeError("Training data not cached. Retrain the model.")

        torch.save(
            {
                "model_type": self.model_type.value,
                "norm_method": engine.norm_method.value,
                "x_columns": engine.x_columns,
                "y_columns": engine.y_columns,
                "num_cells": engine.num_cells,
                "cell_ids": engine.cell_ids,
                "num_features": engine.num_features,
                "x_max_config": self._x_max,
                "x_min_config": self._x_min,
                "normalization": {
                    "xmeans": [s.to_dict() for s in engine.xmeans],
                    "xstds": [s.to_dict() for s in engine.xstds],
                    "xmax": [s.to_dict() for s in engine.xmax],
                    "xmin": [s.to_dict() for s in engine.xmin],
                    "ymeans": [s.to_dict() for s in engine.ymeans],
                    "ystds": [s.to_dict() for s in engine.ystds],
                },
                # ExactGP requires the original training data to build its
                # prediction strategy (caches kernel factorization).
                "training_records": [df.to_dict("records") for df in self._training_data],
                "training_columns": [list(df.columns) for df in self._training_data],
                "model_state_dict": engine.model.state_dict(),
            },
            path,
        )
        logger.info("BayesianDTModel saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Reconstructs the BayesianDigitalTwin from the stored training
        DataFrames (required by ExactGP's prediction strategy), then
        overwrites kernel parameters via ``load_state_dict``.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        x_columns = checkpoint["x_columns"]
        y_columns = checkpoint["y_columns"]
        norm_method = NormMethod(checkpoint["norm_method"])

        # Rebuild training DataFrames from stored records
        training_data = [
            pd.DataFrame(records, columns=cols)
            for records, cols in zip(
                checkpoint["training_records"], checkpoint["training_columns"]
            )
        ]
        self._training_data = training_data

        # Reconstruct BayesianDigitalTwin with the original training data
        # so that ExactGP builds a valid prediction strategy.
        self._engine = BayesianDigitalTwin(
            data_in=training_data,
            x_columns=x_columns,
            y_columns=y_columns,
            norm_method=norm_method,
            x_max=checkpoint["x_max_config"],
            x_min=checkpoint["x_min_config"],
        )

        # Restore trained kernel/mean parameters
        self._engine.model.load_state_dict(checkpoint["model_state_dict"])
        self._engine.model.eval()

        # Restore normalization stats from checkpoint (in case they differ
        # from what was recomputed from the training DataFrames)
        norm = checkpoint["normalization"]
        self._engine.xmeans = [pd.Series(d) for d in norm["xmeans"]]
        self._engine.xstds = [pd.Series(d) for d in norm["xstds"]]
        self._engine.xmax = [pd.Series(d) for d in norm["xmax"]]
        self._engine.xmin = [pd.Series(d) for d in norm["xmin"]]
        self._engine.ymeans = [pd.Series(d) for d in norm["ymeans"]]
        self._engine.ystds = [pd.Series(d) for d in norm["ystds"]]
        self._engine.norm_method = norm_method

        self._norm_method = norm_method
        self._x_max = checkpoint["x_max_config"]
        self._x_min = checkpoint["x_min_config"]
        self._trained = True
        logger.info("BayesianDTModel loaded from %s", path)

    # ---- Bayesian-specific extensions (not in DTModel ABC) ----------------

    @property
    def engine(self) -> BayesianDigitalTwin:
        """Direct access to the underlying :class:`BayesianDigitalTwin`.

        Use this when Bayesian-specific methods are needed (e.g.,
        ``update_trained_gpmodel``, static preprocessing methods).
        Raises ``RuntimeError`` if the model has not been trained yet.
        """
        if self._engine is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._engine

    def update(self, data_in: List[pd.DataFrame]) -> None:
        """Incremental Bayesian update (not in DTModel ABC).

        Delegates to :meth:`BayesianDigitalTwin.update_trained_gpmodel`.
        Requires that ``predict()`` has been called at least once after
        training (engine precondition).
        """
        self.engine.update_trained_gpmodel(data_in)
