from notebooks.MRO_library import *
from notebooks.MRO_library import _check_rlf_threshold
import unittest
import pandas as pd
import numpy as np
import pytest

class TestMROLibrary(unittest.TestCase):
    data = {
    'ue_id': [1,   1,    1,    1,    2,  2,    3,   3,   3],
    'cell_id':    ['A', 'RLF', 'RLF', 'B', 'A', 'A',  'X', 'Y', 'Z'],
    'tick':       [1,   2,    3,    4,    1,  2,    1,   2,   3]
    }
    df = pd.DataFrame(data)

    def test_count_switches(self):
        result = count_switches(self.df)
        self.assertEqual(result, 3)

    def test_count_rlf(self):
        result = count_rlf(self.df)
        self.assertEqual(result, 2)

    def test_calculate_mro_metric(self):
        result = calculate_mro_metric(self.df)
        self.assertEqual(result, 1.85)

    def test_haversine(self):
        lat1, lon1 = -90, -180
        lat2, lon2 = 90, 180
        expected_distance = 20037.508342789242
        result = haversine(lat1, lon1, lat2, lon2)
        self.assertAlmostEqual(result, expected_distance, places=2)

    def test_rlf_threshold(self):
        # Dummy Data
        data_current_tick = pd.DataFrame({
            'ue_id':       [1, 1, 2, 2, 3, 3],
            'cell_id':          [1, 2, 1, 2, 1, 2],
            'cell_rxpower_dbm': [68, 76, 40, 74, 91, 102],
            'sinr_db':          [30, 32, 20, 23, 26, 23],
            'cell_lat':         [10, 20, 30, 40, 50, 60],
            'cell_lon':         [10, 20, 30, 40, 50, 60],
            'cell_carrier_freq_mhz': [20, 20, 20, 20, 20, 20],
            'cell_az_deg':      [10, 20, 30, 40, 50, 60],
            'distance_km':      [10, 20, 30, 40, 50, 60],
            'relative_bearing': [10, 20, 30, 40, 50, 60]
        })

        df = pd.DataFrame({
            'ue_id':       [1, 2, 3],
            'cell_id':          [2, 1, 2],
            'cell_rxpower_dbm': [76, 40, 102],
            'sinr_db':          [32, 20, 23],
            'cell_lat':         [10, 20, 30],
            'cell_lon':         [10, 20, 30],
            'cell_carrier_freq_mhz': [20, 20, 20],
            'cell_az_deg':      [10, 20, 30],
            'distance_km':      [10, 20, 30],
            'relative_bearing': [10, 20, 30]
        })

        result = _check_rlf_threshold(df, data_current_tick, 25)

        expected = pd.DataFrame({
            'ue_id': [1, 2, 3],
            'cell_id': [2, 'RLF', 1],
            'cell_rxpower_dbm': [76.0, -np.inf, 91.0],
            'sinr_db': [32.0, -np.inf, 26.0],
            'cell_lat': [10, 20, 50],
            'cell_lon': [10, 20, 50],
            'cell_carrier_freq_mhz': [20, 20, 20],
            'cell_az_deg': [10, 20, 50],
            'distance_km': [10, 20, 50],
            'relative_bearing': [10, 20, 50]
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

if __name__ == '__main__':
    unittest.main()