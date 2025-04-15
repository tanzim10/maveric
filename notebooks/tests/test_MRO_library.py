from notebooks.MRO_library import *
import unittest
import pandas as pd

class TestMROLibrary(unittest.TestCase):
    data = {
    'mock_ue_id': [1,   1,    1,    1,    2,  2,    3,   3,   3],
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

if __name__ == '__main__':
    unittest.main()