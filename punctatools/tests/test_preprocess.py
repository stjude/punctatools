import os
import shutil
import time
import unittest

import intake_io
import numpy as np
import pandas as pd
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.preprocess import compute_histogram_batch, subtract_background, subtract_background_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'
TMP_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../tmp/' + str(time.time())


@ddt
class TestConversion(unittest.TestCase):

    def test_histogram(self):
        compute_histogram_batch(INPUT_DIR, TMP_DIR + '/hist')
        self.assertEqual(len(os.listdir(TMP_DIR + '/hist')), 2)
        self.assertGreater(len(pd.read_csv(TMP_DIR + '/hist.csv')), 0)
        shutil.rmtree(TMP_DIR)

    @data(
        100,
        [100, 30, 70]
    )
    def test_bg_subtraction_batch(self, bg_value):
        subtract_background_batch(INPUT_DIR, TMP_DIR, bg_value)
        self.assertEqual(len(os.listdir(TMP_DIR)), 2)
        shutil.rmtree(TMP_DIR)

    @data(
        50,
        [10, 30, 70]
    )
    def test_bg_subtraction(self, bg_value):
        img = intake_io.imload(walk_dir(INPUT_DIR)[0])
        bg = np.array([img['image'].data[i].min() for i in range(img['image'].data.shape[0])])
        img = subtract_background(img, bg_value)
        bg_new = np.array([img['image'].data[i].min() for i in range(img['image'].data.shape[0])])
        self.assertSequenceEqual(list(bg - bg_value), list(bg_new))


if __name__ == '__main__':
    unittest.main()
