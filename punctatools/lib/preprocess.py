import os
from typing import Union

import intake_io
import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel
from am_utils.utils import walk_dir
from tqdm import tqdm


def compute_histogram(dataset):
    """
    Compute intensity histogram for a give image.

    Parameters
    ----------
    img : xr.Dataset
        Input image

    Returns
    -------
    pd.DataFrame:
        Histogram as pandas DataFrame
    """
    imghist = pd.DataFrame()
    for i in range(dataset.dims['c']):
        img = dataset.loc[dict(c=dataset.coords['c'][i])]['image'].data
        hist, bins = np.histogram(img, bins=np.max(img) + 1, range=(0, np.max(img) + 1))
        chist = pd.DataFrame({
            'values': bins[:-1],
            'counts': hist
        })
        chist = chist[chist['counts'] > 0]
        chist['channel'] = dataset.coords['c'][i].data
        imghist = pd.concat([imghist, chist], ignore_index=True)
    return imghist


def compute_histogram_batch(input_dir: str, output_dir: str):
    """
    Compute intensity histograms for all images in a folder and save as csv.

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory

    """
    samples = walk_dir(input_dir)
    all_hist = pd.DataFrame()

    for sample in tqdm(samples):
        dataset = intake_io.imload(sample)
        imghist = compute_histogram(dataset)
        imghist['Image name'] = sample
        fn_out = sample.replace(input_dir, output_dir).replace(os.path.splitext(sample)[-1], '.csv')
        os.makedirs(os.path.dirname(fn_out), exist_ok=True)
        imghist.to_csv(fn_out, index=False)
        all_hist = pd.concat([all_hist, imghist], ignore_index=True)
    all_hist.to_csv(output_dir.rstrip('/') + '.csv', index=False)


def subtract_background(dataset, bg_value):
    bg_value = np.array([bg_value]).ravel()
    channels = dataset.coords['c'].data

    if len(bg_value) >= len(channels):
        for i in range(len(channels)):
            img = dataset.loc[dict(c=channels[i])]['image'].data
            img = np.clip(img, bg_value[i], None)
            dataset['image'].loc[dict(c=channels[i])] = img - bg_value[i]
    else:
        img = dataset['image'].data
        img = np.clip(img, bg_value[0], None)
        dataset['image'].data = img - bg_value[0]
    return dataset


def __subtract_bg_helper(item, **kwargs):
    fn_in, fn_out = item
    dataset = intake_io.imload(fn_in)
    dataset = subtract_background(dataset, **kwargs)
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    intake_io.imsave(dataset, fn_out)


def subtract_background_batch(input_dir: str, output_dir: str,
                              bg_value: Union[int, float, list, tuple], n_jobs: int = 8):
    """

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory
    bg_value : scalar or list
        Background values for each channel.
        If one value provided, it will be subtracted from all channels.
    n_jobs : int, optional
        Number of jobs to run in parallel if `parallel` is True
        Default: 8
    """
    run_parallel(items=[(sample,
                         sample.replace(input_dir, output_dir))
                        for sample in walk_dir(input_dir)],
                 process=__subtract_bg_helper,
                 max_threads=n_jobs,
                 bg_value=bg_value)


def rescale_intensity(x, quantiles=(0.0025, 0.9975)):
    mn, mx = [np.percentile(x, p * 100) for p in quantiles]
    if mx > mn + 5:
        return np.clip((x.astype(np.float32) - mn) / (mx - mn), 0, 1)
    else:
        return np.zeros(x.shape, dtype=np.float32)
