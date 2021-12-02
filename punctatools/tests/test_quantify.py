import os
import unittest

import intake_io
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.quantify import quantify
from ..lib.segment import segment_cells, segment_puncta_in_all_channels

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
        dataset = segment_cells(dataset, channel=0, remove_small_mode='2D', add_to_input=True)
        dataset = segment_puncta_in_all_channels(dataset, puncta_channels=puncta_channels,
                                                 minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                                 overlap=1, threshold_detection=0.001, threshold_background=0,
                                                 threshold_segmentation=50, global_background=False,
                                                 segmentation_mode=1, remove_out_of_cell=False)
        cell_stats, puncta_stats = quantify(dataset, channel_names=ch_names,
                                            puncta_channels=puncta_channels)

        self.assertGreater(len(cell_stats), 0)
        self.assertGreater(len(puncta_stats), 0)


if __name__ == '__main__':
    unittest.main()
