import os
import unittest

import intake_io
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.segment import segment_roi, segment_puncta

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'


@ddt
class TestSegmentation(unittest.TestCase):

    @data(
        (0, 0.0003, True),
        (1, 50, True),
        (0, 0.0003, False)
    )
    def test_segment_puncta(self, case):
        segmentation_mode, threshold_segmentation, with_cells = case
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[0])
        if with_cells:
            roi = segment_roi(dataset, channel=0, remove_small_mode='2D')
        else:
            roi = None
        puncta = segment_puncta(dataset, channel=1,
                                roi=roi, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=threshold_segmentation, global_background=False,
                                segmentation_mode=segmentation_mode, maxrad_um=6,
                                remove_out_of_roi=False)
        self.assertGreater(puncta.max(), 0)

    @data(
        (False, 5, 50, True),
        (True, 95, 75, False),
    )
    def test_background_filtering(self, case):
        clear_border, global_background_percentile, background_percentile, global_background = case
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[0])
        roi = segment_roi(dataset, channel=0, remove_small_mode='2D', clear_border=clear_border)
        puncta = segment_puncta(dataset, channel=1,
                                roi=roi, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=0.0001, global_background=global_background,
                                background_percentile=background_percentile,
                                global_background_percentile=global_background_percentile,
                                segmentation_mode=0,
                                remove_out_of_roi=False)
        self.assertGreater(puncta.max(), 0)


if __name__ == '__main__':
    unittest.main()
