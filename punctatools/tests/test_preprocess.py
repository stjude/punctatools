import os
import shutil
import unittest

import intake_io
import numpy as np
import pandas as pd
from am_utils.utils import walk_dir
from ddt import ddt, data

from ..lib.preprocess import compute_histogram_batch, subtract_background, subtract_background_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/stacks'


@ddt
class TestConversion(unittest.TestCase):

    def test_histogram(self):
        dir_out = 'tmp_out'
        compute_histogram_batch(INPUT_DIR, dir_out + '/hist')
        self.assertEqual(len(os.listdir(dir_out + '/hist')), 2)
        self.assertGreater(len(pd.read_csv(dir_out + '/hist.csv')), 0)
        shutil.rmtree(dir_out)

    @data(
        100,
        [100, 30, 70]
    )
    def test_bg_subtraction_batch(self, bg_value):
        dir_out = 'tmp_out'
        subtract_background_batch(INPUT_DIR, dir_out, bg_value)
        self.assertEqual(len(os.listdir(dir_out)), 2)
        shutil.rmtree(dir_out)

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
