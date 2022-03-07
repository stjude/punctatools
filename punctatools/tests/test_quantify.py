import os
import unittest

import numpy as np
import xarray as xr
from ddt import ddt

from ..lib.quantify import quantify

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'
INPUT_DIR_2D = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/slices'


@ddt
class TestQuantify(unittest.TestCase):

    def test_coloc(self):
        img = np.zeros([6, 10, 50, 50])
        img[0] = np.random.randint(0, 100, (10, 50, 50))
        img[1] = img[2] = np.random.randint(0, 100, (10, 50, 50))
        img[3, 0:3, 10:-10, 10:-1] = 1
        img[3, 3:6, 10:-10, 10:-1] = 2
        img[3, 7:, 10:-10, 10:-1] = 3

        img[4, 0:3, 30:35, 30:35] = 2
        img[5, 0:3, 30:35, 30:35] = 4

        img[4, 3:6, 30:36, 30:36] = 1
        img[5, 3:6, 30:36, 33:39] = 3

        img[4, 7:, 30:35, 10:20] = 6
        img[5, 7:, 30:35, 30:35] = 7

        dataset = xr.Dataset(data_vars=dict(image=({'c': 6, 'z': 10, 'y': 50, 'x': 60}, img.astype(np.uint16))))
        for c in ['x', 'y', 'z']:
            dataset.coords[c] = np.arange(dataset.dims[c])

        roi_quant, puncta_quant = quantify(dataset, channel_names=['ch1', 'ch2', 'ch3'],
                                           puncta_channels=[1, 2])
        c = 'Pearson correlation coefficient ch2 vs ch3'
        self.assertAlmostEqual(np.mean(roi_quant[c]), 1, 5)
        self.assertAlmostEqual(np.mean(puncta_quant[c]), 1, 5)

        self.assertSequenceEqual(list(np.round_(roi_quant['Overlap coefficient ch2_ch3_coloc'], 2)),
                                 [1, 0.33, 0])

        for c in ['Pearson correlation coefficient ch1 vs ch2',
                  'Pearson correlation coefficient ch1 vs ch3']:
            self.assertLess(np.mean(np.abs(roi_quant[c])), 0.2)
            self.assertLess(np.mean(np.abs(puncta_quant[c])), 0.2)


if __name__ == '__main__':
    unittest.main()
