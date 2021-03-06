import os
import shutil
import time
import unittest

import intake_io
import numpy as np
from ddt import ddt, data

from ..lib.convert import load_random_dataset, images_to_stacks, check_metadata, get_number_of_stacks

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data/slices'
TMP_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../tmp/' + str(time.time())


@ddt
class TestConversion(unittest.TestCase):

    def test_check_metadata(self):
        dataset, spacing = check_metadata(INPUT_DIR)
        self.assertIsNone(spacing[0])
        self.assertEqual(round(spacing[1], 2), 0.11)
        self.assertEqual(intake_io.get_spacing_units(dataset)[-1], 'µm')
        self.assertEqual(intake_io.get_axes(dataset), 'czyx')
        self.assertSequenceEqual(list(dataset['image'].shape), [3, 5, 326, 326])

    def test_nstacks(self):
        nstacks = get_number_of_stacks(INPUT_DIR)
        self.assertEqual(nstacks, 2)

    def test_random_ds(self):
        dataset = load_random_dataset(INPUT_DIR)
        self.assertIsNone(intake_io.get_spacing(dataset)[0])
        dataset = load_random_dataset(INPUT_DIR, spacing=(0.2, None, None))
        self.assertSequenceEqual(list(np.round(intake_io.get_spacing(dataset), 2)), [0.2, 0.11, 0.11])

    def test_convert(self):
        images_to_stacks(INPUT_DIR, TMP_DIR, spacing=(0.2, None, None), parallel=False)
        files = os.listdir(TMP_DIR)
        self.assertEqual(len(files), 2)
        dataset = intake_io.imload(os.path.join(TMP_DIR, files[0]))
        self.assertSequenceEqual(list(np.round(intake_io.get_spacing(dataset), 2)), [0.2, 0.11, 0.11])
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
