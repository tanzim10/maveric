from notebooks.MRO_library import *
from notebooks.MRO_library import (
    _check_rlf_threshold,
    _check_hyst,
    _check_ttt,
    _perform_attachment_hyst_ttt_per_tick,
    _check_hyst_in_current_tick,
)
import unittest
import pandas as pd
import numpy as np
import os


class TestMROLibrary(unittest.TestCase):
    data = {
        "ue_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
        "cell_id": ["A", "RLF", "RLF", "B", "A", "A", "X", "Y", "Z"],
        "tick": [1, 2, 3, 4, 1, 2, 1, 2, 3],
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

    def test_calculate_received_power(self):
        distance_km = 10
        frequency_mhz = 1800
        expected_power = -94.55545010206613  # Example expected value
        result = calculate_received_power(distance_km, frequency_mhz)
        self.assertAlmostEqual(result, expected_power, places=2)

    def test_concatenate_ue_to_topology(self):
        # Sample UE data
        ue_data = pd.DataFrame(
            {
                "mock_ue_id": [1, 2],
                "latitude": [10.0, 20.0],
                "longitude": [30.0, 40.0],
                "tick": [100, 200],
            }
        )

        # Sample topology data (cells)
        topology = pd.DataFrame(
            {
                "cell_id": [101, 102],
                "cell_lat": [10.5, 20.5],
                "cell_lon": [30.5, 40.5],
                "cell_az_deg": [45, 90],
                "cell_carrier_freq_mhz": [1000, 2000],
            }
        )

        # Call the function with the test data
        result = concatenate_ue_to_topology(ue_data, topology)

        # Expected results (this is a simplified expectation based on the sample haversine)
        expected_result = pd.DataFrame(
            {
                "mock_ue_id": [1, 2],
                "longitude": [30.0, 40.0],
                "latitude": [10.0, 20.0],
                "tick": [100, 200],
                "cell_lat": [10.5, 20.5],
                "cell_lon": [30.5, 40.5],
                "cell_id": [101, 102],
                "cell_az_deg": [45, 90],
                "cell_carrier_freq_mhz": [1000, 2000],
            }
        )
        # Ensure that the data types match for comparison
        result = result.astype(
            {
                "mock_ue_id": "int64",
                "longitude": "float64",
                "latitude": "float64",
                "tick": "int64",
                "cell_lat": "float64",
                "cell_lon": "float64",
                "cell_id": "int64",
                "cell_az_deg": "int64",
                "cell_carrier_freq_mhz": "int64",
            }
        )

        expected_result = expected_result.astype(
            {
                "mock_ue_id": "int64",
                "longitude": "float64",
                "latitude": "float64",
                "tick": "int64",
                "cell_lat": "float64",
                "cell_lon": "float64",
                "cell_id": "int64",
                "cell_az_deg": "int64",
                "cell_carrier_freq_mhz": "int64",
            }
        )

        # Assert that the result matches the expected result
        pd.testing.assert_frame_equal(result, expected_result)
        # Assert that the result matches the expected result
        pd.testing.assert_frame_equal(result, expected_result)

    def test_rlf_threshold(self):
        # Dummy Data
        data_current_tick = pd.DataFrame(
            {
                "ue_id": [1, 1, 2, 2, 3, 3],
                "cell_id": [1, 2, 1, 2, 1, 2],
                "cell_rxpower_dbm": [68, 76, 40, 74, 91, 102],
                "sinr_db": [30, 32, 20, 23, 26, 23],
                "cell_lat": [10, 20, 30, 40, 50, 60],
                "cell_lon": [10, 20, 30, 40, 50, 60],
                "cell_carrier_freq_mhz": [20, 20, 20, 20, 20, 20],
                "cell_az_deg": [10, 20, 30, 40, 50, 60],
                "distance_km": [10, 20, 30, 40, 50, 60],
                "relative_bearing": [10, 20, 30, 40, 50, 60],
            }
        )

        df = pd.DataFrame(
            {
                "ue_id": [1, 2, 3],
                "cell_id": [2, 1, 2],
                "cell_rxpower_dbm": [76, 40, 102],
                "sinr_db": [32, 20, 23],
                "cell_lat": [10, 20, 30],
                "cell_lon": [10, 20, 30],
                "cell_carrier_freq_mhz": [20, 20, 20],
                "cell_az_deg": [10, 20, 30],
                "distance_km": [10, 20, 30],
                "relative_bearing": [10, 20, 30],
            }
        )

        result = _check_rlf_threshold(df, data_current_tick, 25)

        expected = pd.DataFrame(
            {
                "ue_id": [1, 2, 3],
                "cell_id": [2, "RLF", 1],
                "cell_rxpower_dbm": [76.0, -np.inf, 91.0],
                "sinr_db": [32.0, -np.inf, 26.0],
                "cell_lat": [10, 20, 50],
                "cell_lon": [10, 20, 50],
                "cell_carrier_freq_mhz": [20, 20, 20],
                "cell_az_deg": [10, 20, 50],
                "distance_km": [10, 20, 50],
                "relative_bearing": [10, 20, 50],
            }
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_connect_ue_to_all_cells(self):
        # Sample UE data
        ue_data = pd.DataFrame(
            {
                "mock_ue_id": [1, 2],
                "latitude": [10.0, 20.0],
                "longitude": [30.0, 40.0],
                "tick": [100, 200],
            }
        )

        # Sample topology data (cells)
        topology = pd.DataFrame(
            {
                "cell_id": [101, 102],
                "cell_lat": [10.5, 20.5],
                "cell_lon": [30.5, 40.5],
                "cell_az_deg": [45, 90],
                "cell_carrier_freq_mhz": [1000, 2000],
            }
        )

        # Call the function with the test data
        result = connect_ue_to_all_cells(ue_data, topology)

        # Expected results based on a simplified assumption of the haversine distance
        expected_result = pd.DataFrame(
            {
                "mock_ue_id": [1, 1, 2, 2],
                "longitude": [30.0, 30.0, 40.0, 40.0],
                "latitude": [10.0, 10.0, 20.0, 20.0],
                "tick": [100, 100, 200, 200],
                "cell_lat": [10.5, 20.5, 10.5, 20.5],
                "cell_lon": [30.5, 40.5, 30.5, 40.5],
                "cell_id": [101, 102, 101, 102],
                "cell_az_deg": [45, 90, 45, 90],
                "cell_carrier_freq_mhz": [1000, 2000, 1000, 2000],
            }
        )

        # Ensure that the data types match for comparison
        result = result.astype(
            {
                "mock_ue_id": "int64",
                "longitude": "float64",
                "latitude": "float64",
                "tick": "int64",
                "cell_lat": "float64",
                "cell_lon": "float64",
                "cell_id": "int64",
                "cell_az_deg": "int64",
                "cell_carrier_freq_mhz": "int64",
            }
        )

        expected_result = expected_result.astype(
            {
                "mock_ue_id": "int64",
                "longitude": "float64",
                "latitude": "float64",
                "tick": "int64",
                "cell_lat": "float64",
                "cell_lon": "float64",
                "cell_id": "int64",
                "cell_az_deg": "int64",
                "cell_carrier_freq_mhz": "int64",
            }
        )

        # Assert that the result matches the expected result
        pd.testing.assert_frame_equal(result, expected_result)

    def test_add_sinr_column(self):
        data = {
            "ue_id": [1, 1, 2, 2],
            "cell_carrier_freq_mhz": [1000, 1000, 1000, 1000],
            "cell_rxpower_dbm": [
                -90,
                -85,
                -95,
                -92,
            ],  # Received power in dBm for each UE–cell pair
        }
        df = pd.DataFrame(data)

        # Run the function to add the SINR column
        result = add_sinr_column(df)

        # Check if the result has the 'sinr_db' column and if it's computed correctly
        print(result)
        expected_sinr_values = [-5.000001, 4.999996, -3.000007, 2.999986]

        # Check if the SINR values are close to the expected ones
        for idx, expected_sinr in enumerate(expected_sinr_values):
            assert np.isclose(
                result.loc[idx, "sinr_db"], expected_sinr, atol=1e-6
            ), f"Mismatch at index {idx}"

    def test_find_hyst_dff(self):
        data = {
            "cell_rxpower_dbm": [1, 2, -np.inf, -np.inf, 5]
        }
        df = pd.DataFrame(data)
        result = find_hyst_diff(df)
        self.assertEqual(result, 4)

    def test_check_hyst(self):
        input_per_tick = pd.DataFrame({
            'ue_id': [1, 1, 2, 2, 3, 3],
            'cell_id': [1, 2, 1, 2, 1, 2],
            'cell_rxpower_dbm': [-80, -75, -90, -85, -95, -92]
        })
        past_data = pd.DataFrame({
            'ue_id': [1, 2, 3],
            'cell_id_past': [2, 1, 2],  # Past cell_id each UE was attached to
            'cell_rxpower_dbm_past': [-78, -88, -93]  # Past received power levels
        })

        expected_result = pd.DataFrame({
            'ue_id': [1, 2, 3],
            'cell_id': [2, 2, 2],
            'cell_rxpower_dbm': [-75.0, -85.0, -92.0]
        }).sort_values(by='ue_id').reset_index(drop=True)

        # if everything went right ue_id 2 should switch if hyst == 5
        hyst = 5
        result = _check_hyst(input_per_tick, past_data, hyst)
        result = result.sort_values(by='ue_id').reset_index(drop=True)

        # Dataframe called 'Expected' using values from result
        
        result = result.astype({'ue_id': 'float', 'cell_id': 'float', 'cell_rxpower_dbm': 'float'})
        expected_result = expected_result.astype({'ue_id': 'float', 'cell_id': 'float', 'cell_rxpower_dbm': 'float'})

        self.assertTrue(result.equals(expected_result))
        
    def test_check_ttt(self):
        # Example --> _update_current_attachment()

        # consider 3 UEs and 2 cells
        # TTT = 3
        # hyst = 5

        # past_attachment: pd.DataFrame --> output of _update_current_attachment() from last tick, last attached cells
        past_attachment = pd.DataFrame({
            'ue_id':       [1, 2, 3],
            'cell_id':          [2, 2, 1],
            'cell_rxpower_dbm': [70, 75, 90]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        df1 = pd.DataFrame({
            'ue_id':       [1, 2, 3],
            'cell_id':          [1, 1, 2],
            'cell_rxpower_dbm': [80, 75, 90]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        df2 = pd.DataFrame({
            'ue_id':       [1, 2, 3],
            'cell_id':          [2, 1, 2],
            'cell_rxpower_dbm': [75, 76, 100]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        # strongest_server_history: List[pd.DataFrame] --> strongest cell for previous TTT-1 ticks
        strongest_server_history = [df1, df2] # len is TTT-1 = 3-1 = 2

        # ue_data_for_current_tick: pd.DataFrame --> contains calculated rx_power for UEs x cells
        ue_data_for_current_tick = pd.DataFrame({
            'ue_id':       [1, 1, 2, 2, 3, 3],
            'cell_id':          [1, 2, 1, 2, 1, 2],
            'cell_rxpower_dbm': [68, 76, 40, 74, 91, 102]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        current_attachment = _check_ttt(strongest_server_history, ue_data_for_current_tick, past_attachment)
        
        
        # ue 1 didn't switch attachment
        # ue 2 switched attachment
        # ue 3 switched attachment
        
        # Creating expected datasets for current_attachment as it the output
        
        current_attachment_expected = pd.DataFrame({
            'ue_id': [1, 2, 3],
            'cell_id': [2, 1, 2],
            'cell_rxpower_dbm': [76, 40, 102]
        }).astype({
            'ue_id': 'float',
            'cell_id': 'float',
            'cell_rxpower_dbm': 'float'
        })
        (current_attachment).astype({
            'ue_id': 'float',
            'cell_id': 'float',
            'cell_rxpower_dbm': 'float'
        })
        
        pd.testing.assert_frame_equal(current_attachment_expected, current_attachment)

    def test_check_hyst_in_current_tick(self):
        # Example --> _check_hyst_in_current_tick()
        # consider 3 UEs and 2 cells

        TTT = 3
        hyst = 5

        # past_attachment: pd.DataFrame --> output of _update_current_attachment() from last tick, last attached cells
        past_attachment = pd.DataFrame({
            'ue_id':       [1, 2, 3],
            'cell_id':          [2, 2, 1],
            'cell_rxpower_dbm': [70, 75, 90]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        # ue_data_for_current_tick: pd.DataFrame --> contains calculated rx_power for UEs x cells
        ue_data_for_current_tick = pd.DataFrame({
            'ue_id':       [1, 1, 2, 2, 3, 3],
            'cell_id':          [1, 2, 1, 2, 1, 2],
            'cell_rxpower_dbm': [68, 76, 40, 74, 91, 102]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        # this df is calculated inside _update_current_attachment()
        current_attachment = pd.DataFrame({
            "ue_id": [1, 2, 3],
            "cell_id": [2, 1, 2],
            "cell_rxpower_dbm": [76, 40, 102]
        }).astype({
            'ue_id': 'int',
            'cell_id': 'int',
            'cell_rxpower_dbm': 'float'
        })

        # as (mock_ue_id, cell_id) = (2, 1) connects to 40 dbm which doesn't satisfy hyst
        # as previous connection (2, 2) offers 74 dbm, so revert.
        current_attachment = _check_hyst_in_current_tick(ue_data_for_current_tick, current_attachment, past_attachment, hyst).astype({
            'ue_id': 'float',
            'cell_id': 'float',
            'cell_rxpower_dbm': 'float'
        })
        
        # Creating expected data
        
        current_attachment_expected = pd.DataFrame({
            'ue_id': [1, 2, 3],
            'cell_id': [2, 2, 2],
            'cell_rxpower_dbm': [76, 74, 102],
        }).astype({
            'ue_id': 'float',
            'cell_id': 'float',
            'cell_rxpower_dbm': 'float'
        })
        
        pd.testing.assert_frame_equal(current_attachment_expected, current_attachment)
        
if __name__ == "__main__":
    unittest.main()
