import os
import re

import intake_io
import numpy as np
from am_utils.parallel import run_parallel
from am_utils.utils import walk_dir
from tqdm import tqdm


def __get_i_tag(input_dir: str, channel_code: str = "_C", z_position_code: str = "_Z"):
    fn = walk_dir(input_dir)[0]
    codes = [channel_code, z_position_code]
    codes = [code for code in codes if code is not None]
    if len(codes) == 0:
        raise ValueError('Either "channel_code" or "z_position_code" must be not None!')
    ind = np.argmin([re.findall(rf'(.*){code}\d+', fn)[0] for code in codes])
    return re.compile(rf"{input_dir}(.*){codes[ind]}\d+")


def __get_source(input_dir: str, channel_code: str = "_C", z_position_code: str = "_Z", **kwargs):
    tag = {}
    if channel_code is not None:
        tag['c'] = channel_code
    if z_position_code is not None:
        tag['z'] = z_position_code
    tag['i'] = __get_i_tag(input_dir, channel_code=channel_code, z_position_code=z_position_code)
    src = intake_io.source.FilePatternSource(input_dir,
                                             axis_tags=tag,
                                             extensions=[os.path.splitext(walk_dir(input_dir)[0])[-1]],
                                             include_filters=[],
                                             exclude_filters=[],
                                             **kwargs
                                             )
    return src


def __replace_spacing(spacing, spacing_new):
    spacing = np.array(spacing)
    for i in range(len(spacing)):
        if spacing_new[i] is not None:
            spacing[i] = spacing_new[i]
    return tuple(spacing)


def __convert_spacing(spacing, axes):
    sp_dict = dict()
    for ax, sp in zip(axes[-len(spacing):], spacing):
        if sp is not None:
            sp_dict[ax] = sp
    return sp_dict


def __get_source_with_metadata(input_dir: str, channel_code: str = "_C", z_position_code: str = "_Z",
                               spacing: tuple = (None, None, None)):
    src = __get_source(input_dir, channel_code, z_position_code)
    dataset = intake_io.imload(src, partition=0)
    spacing2 = intake_io.get_spacing(dataset)
    spacing = __replace_spacing(spacing2, spacing)
    spacing = __convert_spacing(spacing, intake_io.get_axes(dataset))
    src = __get_source(input_dir, channel_code, z_position_code, metadata={"spacing": spacing})
    return src


def __convert_helper(item, **kwargs):
    __convert_one_image(partition=item, **kwargs)


def __convert_one_image(src, partition, output_dir):
    dataset = intake_io.imload(src, partition=partition)
    fn_out = str(intake_io.imload(src, metadata_only=True)['metadata']['coords']['i'][partition].strip('/'))
    fn_out = os.path.join(output_dir, fn_out.replace(' ', '_') + '.tif')
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    try:
        intake_io.imsave(dataset, fn_out)
    except UnicodeDecodeError:
        dataset['image'].metadata['spacing_units'] = {}
        intake_io.imsave(dataset, fn_out)


def check_metadata(input_dir: str, channel_code: str = "_C", z_position_code: str = "_Z"):
    """

    Parameters
    ----------
    input_dir : str
        Input directory
    channel_code : str
        Sequence of characters that precedes the channel numbering.
        Default: "_C"
    z_position_code : str
        Sequence of characters that precedes the z layer numbering.
        Default: "_Z"

    Returns
    -------
    dataset : xr.Dataset
        Example image stack as xarray dataset.
    spacing : tuple
        List of spacing values along z, y, x axes.
    """
    src = __get_source(input_dir, channel_code, z_position_code)
    dataset = intake_io.imload(src, partition=0)
    spacing = intake_io.get_spacing(dataset)
    return dataset, spacing


def get_number_of_stacks(input_dir: str, channel_code: str = "_C", z_position_code: str = "_Z"):
    """
    Return the number of stacks in the dataset.

    Parameters
    ----------
    input_dir : str
        Input directory
    channel_code : str
        Sequence of characters that precedes the channel numbering.
        Default: "_C"
    z_position_code : str
        Sequence of characters that precedes the z layer numbering.
        Default: "_Z"

    Returns
    -------
    n_stacks : int
        Number of stacks in the dataset.

    """
    src = __get_source(input_dir, channel_code, z_position_code)
    n_stacks = intake_io.imload(src, metadata_only=True)['npartitions']
    return n_stacks


def load_random_dataset(input_dir: str,
                        channel_code: str = "_C", z_position_code: str = "_Z",
                        spacing: tuple = (None, None, None)):
    """
    Loads a random image stack.

    Parameters
    ----------
    input_dir : str
        Input directory
    channel_code : str
        Sequence of characters that precedes the channel numbering.
        Default: "_C"
    z_position_code : str
        Sequence of characters that precedes the z layer numbering.
        Default: "_Z"
    spacing : tuple
        List of spacing values along z, y, x axes.
        Set corresponding values to `None` if the value should be loaded from the image file.
        Run `check_metadata` to check if the saved spacing is correct.

    Returns
    -------
    dataset : xr.Dataset
        Example image stack as xarray dataset.

    """
    src = __get_source_with_metadata(input_dir, channel_code, z_position_code, spacing)
    npartitions = intake_io.imload(src, metadata_only=True)['npartitions']
    dataset = intake_io.imload(src, partition=np.random.randint(npartitions))
    return dataset


def images_to_stacks(input_dir: str, output_dir: str,
                     channel_code: str = "_C", z_position_code: str = "_Z",
                     spacing: tuple = (0.2, None, None),
                     parallel: bool = False, n_jobs: int = 8, **kwargs):
    """

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory
    channel_code : str
        Sequence of characters that precedes the channel numbering.
        Default: "_C"
    z_position_code : str
        Sequence of characters that precedes the z layer numbering.
        Default: "_z"
    spacing : tuple
        List of spacing values along z, y, x axes.
        Set corresponding values to `None` if the value should be loaded from the image file.
        Run `check_metadata` to check if the saved spacing is correct.
    parallel : bool, optional
        If True, run the conversion in parallel.
        Default: False
    n_jobs : int, optional
        Number of jobs to run in parallel if `parallel` is True
        Default: 8

    Returns
    -------

    """
    src = __get_source_with_metadata(input_dir, channel_code, z_position_code, spacing)
    npartitions = intake_io.imload(src, metadata_only=True)['npartitions']

    if parallel:
        run_parallel(items=np.arange(npartitions), process=__convert_helper, max_threads=n_jobs,
                     src=src, output_dir=output_dir, **kwargs)
    else:
        for i in tqdm(range(npartitions)):
            __convert_one_image(src, i, output_dir)
