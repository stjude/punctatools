import os
import shutil
import unittest

import intake_io
from am_utils.utils import walk_dir
from ddt import ddt

from ..lib.segment import segment_cells, segment_puncta, segment_puncta_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/slices'


@ddt
class TestSegmentation(unittest.TestCase):

    def test_segment_cells(self):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_cells(img, remove_small_mode='2D')
        self.assertGreater(mask.max(), 0)

    def test_segment_cells_clear_border(self):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_cells(img, remove_small_mode='2D', clear_border=True)
        self.assertGreater(mask.max(), 0)

    def test_segment_puncta(self):
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[1])
        cells = segment_cells(dataset, channel=0, remove_small_mode='2D')
        puncta = segment_puncta(dataset, channel=0,
                                cells=cells, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=50, global_background=False,
                                segmentation_mode=1, maxrad_um=6,
                                remove_out_of_cell=False)
        self.assertGreater(puncta.max(), 0)

    def test_segment_puncta_batch(self):
        segment_puncta_batch(INPUT_DIR, 'tmp_out/puncta', puncta_channels=[0],
                             parallel=True,
                             minsize_um=0.2, maxsize_um=2, num_sigma=5,
                             overlap=1, threshold_detection=0.001, threshold_background=3,
                             threshold_segmentation=0.0003,
                             segmentation_mode=0)
        shutil.rmtree('tmp_out')


if __name__ == '__main__':
    unittest.main()
