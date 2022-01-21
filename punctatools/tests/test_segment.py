import os
import shutil
import unittest

import intake_io
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.segment import segment_cells, segment_puncta, segment_puncta_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'


@ddt
class TestSegmentation(unittest.TestCase):

    def test_segment_cells(self):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_cells(img, channel=0, remove_small_mode='2D')
        self.assertGreater(mask.max(), 0)

    def test_segment_cells_clear_border(self):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_cells(img, channel=0, remove_small_mode='2D', clear_border=True)
        self.assertGreater(mask.max(), 0)

    def test_segment_cells_3D(self):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        mask = segment_cells(img, channel=0, remove_small_mode='2D', do_3D=True, gpu=False)
        self.assertGreater(mask.max(), 0)
        self.assertEqual(len(mask.shape), 3)

    @data(
        (0, 0.0003, True),
        (1, 50, True),
        (2, 3, True),
        (0, 0.0003, False),
        (1, 50, False),
        (2, 3, False)
    )
    def test_segment_puncta(self, case):
        segmentation_mode, threshold_segmentation, with_cells = case
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[0])
        if with_cells:
            cells = segment_cells(dataset, channel=0, remove_small_mode='2D')
        else:
            cells = None
        puncta = segment_puncta(dataset, channel=1,
                                cells=cells, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=threshold_segmentation, global_background=False,
                                segmentation_mode=segmentation_mode, maxrad_um=6,
                                remove_out_of_cell=False)
        self.assertGreater(puncta.max(), 0)

    @data(
        (95, 75, True),
        (95, 50, True),
        (5, 50, True),
        (95, 75, False),
        (95, 50, False),
        (5, 50, False),
    )
    def test_background_filtering(self, case):
        global_background_percentile, background_percentile, global_background = case
        dataset = intake_io.imload(walk_dir(INPUT_DIR)[0])
        cells = segment_cells(dataset, channel=0, remove_small_mode='2D')
        puncta = segment_puncta(dataset, channel=1,
                                cells=cells, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                                overlap=1, threshold_detection=0.001, threshold_background=0,
                                threshold_segmentation=0.0001, global_background=global_background,
                                background_percentile=background_percentile,
                                global_background_percentile=global_background_percentile,
                                segmentation_mode=0,
                                remove_out_of_cell=False)
        self.assertGreater(puncta.max(), 0)

    def test_segment_puncta_batch(self):
        segment_puncta_batch(INPUT_DIR, 'tmp_out/puncta', puncta_channels=[1, 2],
                             parallel=True,
                             minsize_um=0.2, maxsize_um=2, num_sigma=5,
                             overlap=1, threshold_detection=0.001, threshold_background=3,
                             threshold_segmentation=[0.0003, 50],
                             segmentation_mode=[0, 1])
        shutil.rmtree('tmp_out')


if __name__ == '__main__':
    unittest.main()
