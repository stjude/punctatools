import os
import shutil
import time
import unittest

import pandas as pd
from am_utils.utils import walk_dir
from ddt import ddt

from ..lib.quantify import quantify_batch
from ..lib.segment import segment_roi_batch, segment_puncta_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'
TMP_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../tmp/' + str(time.time())


@ddt
class TestBatch(unittest.TestCase):

    def test_analyze_batch(self):
        segment_roi_batch(INPUT_DIR, TMP_DIR + '/roi', channel=0,
                          remove_small_mode='2D',
                          clear_border=True)
        segment_puncta_batch(TMP_DIR + '/roi', TMP_DIR + '/puncta', puncta_channels=[1, 2],
                             parallel=False,
                             minsize_um=0.2, maxsize_um=2, num_sigma=5,
                             overlap=1, threshold_detection=0.001, threshold_background=0,
                             threshold_segmentation=50,
                             segmentation_mode=1, global_background=False)
        quantify_batch(TMP_DIR + '/puncta', TMP_DIR + '/roiquant', TMP_DIR + '/punctaquant',
                       channel_names=['DNA', 'GFP', 'mCherry'],
                       puncta_channels=[1, 2]
                       )

        self.assertEqual(len(walk_dir(TMP_DIR)), 10)
        for fn in [TMP_DIR + '/punctaquant.csv', TMP_DIR + '/roiquant.csv']:
            df = pd.read_csv(fn)
            self.assertGreater(len(df), 0)
        shutil.rmtree(TMP_DIR)


if __name__ == '__main__':
    unittest.main()
