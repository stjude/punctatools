import os
import unittest

import intake_io
import numpy as np
import xarray as xr
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.quantify import quantify
from ..lib.segment import segment_roi, segment_puncta_in_all_channels

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'
INPUT_DIR_2D = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/slices'


@ddt
class TestQuantify(unittest.TestCase):

    @data(
        (INPUT_DIR, 0, ['DNA', 'GFP', 'mCherry'], [1, 2]),
        (INPUT_DIR_2D, 1, ['ch0'], [0])
    )
    def test_quantify(self, case):
        input_dir, ind, ch_names, puncta_channels = case
        dataset = intake_io.imload(walk_dir(input_dir)[ind])
        dataset = segment_roi(dataset, channel=0, remove_small_mode='2D', add_to_input=True)
        dataset = segment_puncta_in_all_channels(dataset, puncta_channels=puncta_channels,
                                                 minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                                 overlap=1, threshold_detection=0.001, threshold_background=0,
                                                 threshold_segmentation=50, global_background=False,
                                                 segmentation_mode=1, remove_out_of_cell=False)
        cell_stats, puncta_stats = quantify(dataset, channel_names=ch_names,
                                            puncta_channels=puncta_channels)

        self.assertGreater(len(cell_stats), 0)
        self.assertGreater(len(puncta_stats), 0)

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

        cell_stats, puncta_stats = quantify(dataset, channel_names=['ch1', 'ch2', 'ch3'],
                                            puncta_channels=[1, 2])
        c = 'Pearson correlation coefficient ch2 vs ch3'
        self.assertAlmostEqual(np.mean(cell_stats[c]), 1, 5)
        self.assertAlmostEqual(np.mean(puncta_stats[c]), 1, 5)

        self.assertSequenceEqual(list(np.round_(cell_stats['Overlap coefficient ch2_ch3_coloc'], 2)),
                                 [1, 0.33, 0])

        for c in ['Pearson correlation coefficient ch1 vs ch2',
                  'Pearson correlation coefficient ch1 vs ch3']:
            self.assertLess(np.mean(np.abs(cell_stats[c])), 0.1)
            self.assertLess(np.mean(np.abs(puncta_stats[c])), 0.1)


if __name__ == '__main__':
    unittest.main()
