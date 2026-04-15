# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DTModel contract tests.

Every concrete DTModel implementation must pass every test in
:class:`DTModelContractTests`. Adding a new algorithm only requires creating
a subclass of DTModelContractTests and setting up ``self.model`` in
``setUp()``.
"""

import os
import tempfile
import unittest

import numpy as np

from radp.digital_twin.rf.base_model import DTModel, DTModelType, TrainConfig
from radp.digital_twin.rf.bayesian.bayesian_wrapper import BayesianDTModel, BayesianTrainConfig
from radp.digital_twin.rf.tests.test_fixtures import (
    build_data_in_lists,
    prepare_standard_data,
)
from radp.digital_twin.utils import constants
from radp.utility.simulation_utils import seed_everything

# Feature columns used across all contract tests
_X_COLUMNS = [constants.CELL_EL_DEG, constants.LOG_DISTANCE, constants.RELATIVE_BEARING]
_Y_COLUMNS = ["avg_rsrp"]

# Minimal training config so tests finish quickly
_FAST_CONFIG = TrainConfig(max_iter=5, learning_rate=0.05, stopping_threshold=1e-5)


class DTModelContractTests:
    """Mixin that defines the DTModel contract.

    Subclasses must define in ``setUp()``:
      - ``self.model``       : untrained DTModel instance
      - ``self.data_in``     : List[pd.DataFrame] for training
      - ``self.predict_dfs`` : List[pd.DataFrame] for prediction
      - ``self.x_columns``   : feature column list
      - ``self.y_columns``   : target column list
      - ``self.config``      : TrainConfig (fast, for tests)

    Subclasses may also override ``make_fresh_model()`` to return a new
    untrained instance with the same configuration.
    """

    # ---- helpers ----------------------------------------------------------

    def make_fresh_model(self) -> DTModel:
        raise NotImplementedError("Subclass must override make_fresh_model()")

    def _train(self):
        return self.model.train(
            self.data_in,
            self.x_columns,
            self.y_columns,
            self.config,
        )

    # ---- contract tests ---------------------------------------------------

    def test_model_type_is_enum(self):
        self.assertIsInstance(self.model.model_type, DTModelType)

    def test_is_trained_false_before_training(self):
        fresh = self.make_fresh_model()
        self.assertFalse(fresh.is_trained)

    def test_train_returns_loss_array(self):
        loss = self._train()
        self.assertIsInstance(loss, np.ndarray)
        self.assertGreater(loss.size, 0)

    def test_is_trained_true_after_training(self):
        self._train()
        self.assertTrue(self.model.is_trained)

    def test_predict_returns_tuple_of_two_arrays(self):
        self._train()
        result = self.model.predict(self.predict_dfs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        means, stds = result
        self.assertIsInstance(means, np.ndarray)
        self.assertIsInstance(stds, np.ndarray)

    def test_predict_means_shape(self):
        self._train()
        means, _ = self.model.predict(self.predict_dfs)
        n_cells = len(self.predict_dfs)
        n_locs = self.predict_dfs[0].shape[0]
        self.assertEqual(means.shape, (n_cells, n_locs))

    def test_predict_uncertainty_nonnegative(self):
        self._train()
        _, stds = self.model.predict(self.predict_dfs)
        self.assertTrue(np.all(stds >= 0))

    def test_predict_means_finite(self):
        self._train()
        means, _ = self.model.predict(self.predict_dfs)
        self.assertTrue(np.all(np.isfinite(means)))

    def test_save_load_roundtrip(self):
        self._train()
        means_before, _ = self.model.predict(self.predict_dfs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            self.model.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded = self.make_fresh_model()
            loaded.load(tmp_path)
            self.assertTrue(loaded.is_trained)

            means_after, _ = loaded.predict(self.predict_dfs)
            np.testing.assert_allclose(means_before, means_after, rtol=1e-4)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# BayesianDTModel contract
# ---------------------------------------------------------------------------

class TestBayesianDTModelContract(DTModelContractTests, unittest.TestCase):
    """BayesianDTModel (wrapper) must satisfy the full DTModel contract."""

    @classmethod
    def setUpClass(cls):
        seed_everything(1)

    def setUp(self):
        train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)

        self.x_columns = _X_COLUMNS
        self.y_columns = _Y_COLUMNS
        self.data_in = build_data_in_lists(train_map)
        self.predict_dfs = build_data_in_lists(test_map)
        self.config = BayesianTrainConfig(
            max_iter=5, learning_rate=0.05, stopping_threshold=1e-5
        )
        self.model = BayesianDTModel(x_max=x_max, x_min=x_min)

    def make_fresh_model(self) -> DTModel:
        _, _, _, x_max, x_min = prepare_standard_data(test_size=0.4)
        return BayesianDTModel(x_max=x_max, x_min=x_min)


# ---------------------------------------------------------------------------
# SVGPDigitalTwin contract — added after SVGP engine is implemented
# ---------------------------------------------------------------------------

try:
    from radp.digital_twin.rf.svgp.svgp_engine import SVGPDigitalTwin, SVGPTrainConfig

    class TestSVGPDTModelContract(DTModelContractTests, unittest.TestCase):
        """SVGPDigitalTwin must satisfy the full DTModel contract."""

        @classmethod
        def setUpClass(cls):
            seed_everything(1)

        def setUp(self):
            train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)

            self.x_columns = _X_COLUMNS
            self.y_columns = _Y_COLUMNS
            self.data_in = build_data_in_lists(train_map)
            self.predict_dfs = build_data_in_lists(test_map)
            self.config = SVGPTrainConfig(
                max_iter=5,
                learning_rate=0.05,
                stopping_threshold=1e-5,
                num_inducing_points=4,
                batch_size=4,
                num_epochs=3,
            )
            self.model = SVGPDigitalTwin(x_max=x_max, x_min=x_min)

        def make_fresh_model(self) -> DTModel:
            _, _, _, x_max, x_min = prepare_standard_data(test_size=0.4)
            return SVGPDigitalTwin(x_max=x_max, x_min=x_min)

except ImportError:
    pass  # SVGP tests activate once svgp_engine.py exists


if __name__ == "__main__":
    unittest.main()
