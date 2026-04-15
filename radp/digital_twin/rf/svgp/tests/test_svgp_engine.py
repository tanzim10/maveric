# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for SVGPDigitalTwin.

Tests cover:
  - End-to-end train + predict on small sample data
  - Output shapes
  - Inducing point selection (k-means and random)
  - Mini-batch DataLoader integration
  - Save/load roundtrip
  - Reproducibility with fixed seed
  - Training speed comparison vs Bayesian (benchmark, optional)
"""

import os
import tempfile
import time
import unittest

import gpytorch
import numpy as np
import torch

from radp.digital_twin.utils import constants
from radp.digital_twin.rf.svgp.svgp_engine import (
    SVGPDigitalTwin,
    SVGPGPModel,
    SVGPTrainConfig,
    _kmeans_inducing_points,
    _select_inducing_points,
)
from radp.digital_twin.rf.tests.test_fixtures import (
    augment_ue_data,
    build_data_in_lists,
    generate_synthetic_rf_data,
    get_sample_site_config_and_ue_data,
    get_x_max_and_x_min,
    prepare_standard_data,
    split_training_and_test_data,
)
from radp.utility.simulation_utils import seed_everything

# ------------------------------------------------------------------
# Small fast config used across most tests
# ------------------------------------------------------------------
_FAST_CONFIG = SVGPTrainConfig(
    num_inducing_points=4,
    batch_size=4,
    num_epochs=5,
    learning_rate=0.05,
    stopping_threshold=1e-5,
)

_X_COLUMNS = [constants.CELL_EL_DEG, constants.LOG_DISTANCE, constants.RELATIVE_BEARING]
_Y_COLUMNS = ["avg_rsrp"]


class TestSVGPEndToEnd(unittest.TestCase):
    """End-to-end train → predict pipeline on small sample data."""

    @classmethod
    def setUpClass(cls):
        seed_everything(1)
        train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)
        cls.train_list = build_data_in_lists(train_map)
        cls.test_list = build_data_in_lists(test_map)
        cls.x_max = x_max
        cls.x_min = x_min

    def _make_model(self):
        return SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)

    def test_train_returns_2d_loss_array(self):
        model = self._make_model()
        loss = model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        self.assertIsInstance(loss, np.ndarray)
        self.assertEqual(loss.ndim, 2)
        n_cells = len(self.train_list)
        self.assertEqual(loss.shape[0], n_cells)
        self.assertGreater(loss.shape[1], 0)

    def test_is_trained_after_train(self):
        model = self._make_model()
        self.assertFalse(model.is_trained)
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        self.assertTrue(model.is_trained)

    def test_predict_returns_tuple(self):
        model = self._make_model()
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        result = model.predict(self.test_list)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_predict_shape(self):
        model = self._make_model()
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        means, stds = model.predict(self.test_list)
        n_cells = len(self.test_list)
        n_locs = self.test_list[0].shape[0]
        self.assertEqual(means.shape, (n_cells, n_locs))
        self.assertEqual(stds.shape, (n_cells, n_locs))

    def test_predict_means_finite(self):
        model = self._make_model()
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        means, _ = model.predict(self.test_list)
        self.assertTrue(np.all(np.isfinite(means)))

    def test_predict_uncertainty_nonnegative(self):
        model = self._make_model()
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        _, stds = model.predict(self.test_list)
        self.assertTrue(np.all(stds >= 0))

    def test_predict_mutates_dataframe(self):
        model = self._make_model()
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        model.predict(self.test_list)
        for df in self.test_list:
            self.assertIn(constants.RXPOWER_DBM, df.columns)
            self.assertIn(constants.RXPOWER_STDDEV_DBM, df.columns)

    def test_predict_before_train_raises(self):
        model = self._make_model()
        with self.assertRaises(RuntimeError):
            model.predict(self.test_list)

    def test_model_type(self):
        from radp.digital_twin.rf.base_model import DTModelType
        model = self._make_model()
        self.assertEqual(model.model_type, DTModelType.SVGP)


class TestSVGPSaveLoad(unittest.TestCase):
    """Save/load roundtrip tests."""

    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)
        cls.train_list = build_data_in_lists(train_map)
        cls.test_list = build_data_in_lists(test_map)
        cls.x_max = x_max
        cls.x_min = x_min

    def test_save_creates_file(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_save_before_train_raises(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        with self.assertRaises(RuntimeError):
            model.save("/tmp/should_not_exist.pt")

    def test_load_roundtrip_predictions_close(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)

        import copy
        test_dfs_copy = [df.copy() for df in self.test_list]
        means_before, _ = model.predict(test_dfs_copy)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)

            loaded = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
            loaded.load(path)
            self.assertTrue(loaded.is_trained)

            test_dfs_copy2 = [df.copy() for df in self.test_list]
            means_after, _ = loaded.predict(test_dfs_copy2)

            np.testing.assert_allclose(means_before, means_after, rtol=1e-4)
        finally:
            os.unlink(path)

    def test_load_restores_normalization(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = SVGPDigitalTwin()
            loaded.load(path)
            self.assertEqual(loaded._num_cells, model._num_cells)
            self.assertEqual(loaded._cell_ids, model._cell_ids)
            self.assertEqual(loaded._x_columns, _X_COLUMNS)
        finally:
            os.unlink(path)


class TestInducingPoints(unittest.TestCase):
    """Tests for inducing point selection utilities."""

    def test_kmeans_returns_k_points(self):
        X = torch.randn(100, 3)
        centroids = _kmeans_inducing_points(X, k=10)
        self.assertEqual(centroids.shape, (10, 3))

    def test_kmeans_fewer_data_than_k(self):
        X = torch.randn(5, 3)
        centroids = _kmeans_inducing_points(X, k=10)
        self.assertEqual(centroids.shape, (5, 3))

    def test_kmeans_centroids_are_finite(self):
        X = torch.randn(200, 5)
        centroids = _kmeans_inducing_points(X, k=20)
        self.assertTrue(torch.all(torch.isfinite(centroids)))

    def test_select_inducing_kmeans(self):
        X = torch.randn(100, 3)
        pts = _select_inducing_points(X, num_inducing=10, method="kmeans")
        self.assertEqual(pts.shape, (10, 3))

    def test_select_inducing_random(self):
        X = torch.randn(100, 3)
        pts = _select_inducing_points(X, num_inducing=10, method="random")
        self.assertEqual(pts.shape, (10, 3))

    def test_select_inducing_all_when_small(self):
        X = torch.randn(5, 3)
        pts = _select_inducing_points(X, num_inducing=10, method="kmeans")
        self.assertEqual(pts.shape, (5, 3))

    def test_kmeans_deterministic_with_seed(self):
        # Generate X once; seed must be set immediately before k-means so
        # torch.randperm inside both calls uses the same initial state.
        X = torch.randn(50, 4)
        torch.manual_seed(42)
        c1 = _kmeans_inducing_points(X, k=8)
        torch.manual_seed(42)
        c2 = _kmeans_inducing_points(X, k=8)
        self.assertTrue(torch.allclose(c1, c2))


class TestSVGPGPModel(unittest.TestCase):
    """Unit tests for the SVGPGPModel GPyTorch model."""

    def test_forward_returns_mvn(self):
        inducing = torch.randn(10, 3)
        model = SVGPGPModel(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.train()
        likelihood.train()

        X = torch.randn(20, 3)
        output = model(X)
        self.assertIsInstance(output, gpytorch.distributions.MultivariateNormal)

    def test_eval_mode_prediction(self):
        inducing = torch.randn(10, 3)
        model = SVGPGPModel(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.eval()
        likelihood.eval()

        X = torch.randn(5, 3)
        with torch.no_grad():
            pred = likelihood(model(X))
        self.assertEqual(pred.mean.shape, (5,))
        self.assertEqual(pred.variance.shape, (5,))


class TestSVGPReproducibility(unittest.TestCase):
    """Training with the same seed produces identical results."""

    @classmethod
    def setUpClass(cls):
        train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)
        cls.train_list = build_data_in_lists(train_map)
        cls.test_list = build_data_in_lists(test_map)
        cls.x_max = x_max
        cls.x_min = x_min

    def _train_and_predict(self, seed: int):
        seed_everything(seed)
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, _FAST_CONFIG)
        test_dfs = [df.copy() for df in self.test_list]
        means, stds = model.predict(test_dfs)
        return means, stds

    def test_same_seed_same_predictions(self):
        means1, stds1 = self._train_and_predict(seed=7)
        means2, stds2 = self._train_and_predict(seed=7)
        np.testing.assert_array_equal(means1, means2)
        np.testing.assert_array_equal(stds1, stds2)


class TestSVGPTrainingConfig(unittest.TestCase):
    """Config handling and fallback behaviour."""

    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        train_map, test_map, _, x_max, x_min = prepare_standard_data(test_size=0.4)
        cls.train_list = build_data_in_lists(train_map)
        cls.test_list = build_data_in_lists(test_map)
        cls.x_max = x_max
        cls.x_min = x_min

    def test_none_config_uses_defaults(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        # Use a very small epoch override via the base TrainConfig
        from radp.digital_twin.rf.base_model import TrainConfig
        base_cfg = TrainConfig(max_iter=2, learning_rate=0.01, stopping_threshold=1e-5)
        # Should not raise; SVGP wraps this in SVGPTrainConfig internally
        loss = model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, base_cfg)
        self.assertIsNotNone(loss)

    def test_svgp_config_fields_respected(self):
        model = SVGPDigitalTwin(x_max=self.x_max, x_min=self.x_min)
        config = SVGPTrainConfig(num_epochs=2, num_inducing_points=3, batch_size=3)
        loss = model.train(self.train_list, _X_COLUMNS, _Y_COLUMNS, config)
        # With 2 epochs, loss should have at most 2 entries per cell
        self.assertLessEqual(loss.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
