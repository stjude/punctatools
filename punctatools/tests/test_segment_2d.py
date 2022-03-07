import os
import unittest

import intake_io
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.segment import segment_roi, segment_puncta

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/slices'


@ddt
class TestSegmentation(unittest.TestCase):

    @data(
        True, False
    )
    def test_segment_cells(self, clear_border):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_roi(img, remove_small_mode='2D', clear_border=clear_border)
        self.assertGreater(mask.max(), 0)

    def test_segment_puncta(self):
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[1])
        roi = segment_roi(dataset, channel=0, remove_small_mode='2D')
        puncta = segment_puncta(dataset, channel=0,
                                roi=roi, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=50, global_background=False,
                                segmentation_mode=1, maxrad_um=6,
                                remove_out_of_roi=False)
        self.assertGreater(puncta.max(), 0)


if __name__ == '__main__':
    unittest.main()
