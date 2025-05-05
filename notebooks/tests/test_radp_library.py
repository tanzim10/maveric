from notebooks.radp_library import calculate_received_power
import unittest


class TestMobilityRobustnessOptimization(unittest.TestCase):
    def test_calculate_received_power(self):
        dummy_distance = 1
        dummy_freq = 1800
        expected_power = -74.55545010206612
        power = calculate_received_power(
            distance_km=dummy_distance, frequency_mhz=dummy_freq
        )
        self.assertEqual(expected_power, power)
