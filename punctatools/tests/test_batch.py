import os
import shutil
import unittest

from am_utils.utils import walk_dir
from ddt import ddt

from ..lib.quantify import quantify_batch
from ..lib.segment import segment_cells_batch, segment_puncta_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'


@ddt
class TestBatch(unittest.TestCase):

    def test_analyze_batch(self):
        segment_cells_batch(INPUT_DIR, 'tmp_out/cells', channel=0,
                            remove_small_mode='2D',
                            clear_border=True)
        segment_puncta_batch('tmp_out/cells', 'tmp_out/puncta', puncta_channels=[1, 2],
                             parallel=False,
                             minsize_um=0.2, maxsize_um=2, num_sigma=5,
                             overlap=1, threshold_detection=0.001, threshold_background=0,
                             threshold_segmentation=50,
                             segmentation_mode=1, global_background=False)
        quantify_batch('tmp_out/puncta', 'tmp_out/cellstats', 'tmp_out/punctastats',
                       channel_names=['DNA', 'GFP', 'mCherry'],
                       puncta_channels=[1, 2]
                       )

        self.assertEqual(len(walk_dir('tmp_out')), 10)
        shutil.rmtree('tmp_out')


if __name__ == '__main__':
    unittest.main()
