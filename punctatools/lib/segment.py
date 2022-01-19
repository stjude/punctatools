import os

import intake_io
import numpy as np
import xarray as xr
from am_utils.parallel import run_parallel
from am_utils.utils import walk_dir
from cellpose import models
from scipy import ndimage
from skimage import filters
from skimage.feature import blob_log
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border as sk_clear_border
from skimage.segmentation import watershed
from tqdm import tqdm

from .preprocess import rescale_intensity


def __get_images(dataset, do_3D, channel):
    if 'c' in dataset.dims:
        if channel is not None:
            ch_names = dataset.coords['c'].data
            imgs = dataset.loc[dict(c=ch_names[channel])]['image'].data
        else:
            raise ValueError("The image has multiples channels. Provide channel to segment.")
    else:
        imgs = dataset['image'].data
    if 'z' not in dataset.dims:
        imgs = [imgs]
    if do_3D:
        spacing = intake_io.get_spacing(dataset)
        anisotropy = spacing[0] / spacing[-1]
    else:
        anisotropy = None
    return imgs, anisotropy


def __reshape_output(masks, dataset):
    if 'z' not in dataset.dims:
        masks = masks[0]
    else:
        masks = np.array(masks)
    return masks


def __combine_3D(masks, do_3D, diameter,
                 remove_small_mode='3D', remove_small_diam_fraction=0.5,
                 clear_border=False):
    if do_3D is False and len(masks.shape) > 2:

        area = (masks > 0).sum(-1).sum(-1)
        if len(area) > 21:
            ind = np.argmax(area[10:-10]) + 10
        else:
            ind = np.argmax(area)
        minrad = diameter / 2 * remove_small_diam_fraction
        labels = masks[ind:ind + 1].copy()
        if clear_border:
            labels = np.expand_dims(sk_clear_border(labels.max(0)), 0)
        masks = ndimage.median_filter(masks > 0, 3)
        if remove_small_mode == '3D':
            masks = masks * labels
            minvol = 4. / 3 * np.pi * minrad ** 3
            masks = remove_small_objects(masks, min_size=minvol)
        elif remove_small_mode == '2D':
            minarea = np.pi * minrad ** 2
            labels = remove_small_objects(labels, min_size=minarea)
            masks = masks * labels
        else:
            raise ValueError("Invalid value for 'remove_small_mode', must be '3D' or '2D'")
    return masks


def segment_cells(dataset, channel=None, do_3D=False,
                  gpu=True, model_type='cyto',
                  channels=None, diameter=None,
                  remove_small_mode='3D', remove_small_diam_fraction=0.5,
                  clear_border=False, add_to_input=False,
                  return_cellpose_debug=False,
                  **cellpose_kwargs):
    """
    Segment cells/nuclei in one image using cellpose.

    Parameters
    ----------
    dataset : xr.Dataset
        Image in the form of an xarrray dataset (read with intake_io).
    channel : int, optional
        Channel number to use for segmentation, starting from 0.
        If the image has only one channel, this can be left out.
    do_3D : bool, optional
        If True, segment in the 3D mode with cellpose (computationally expensive).
        If False, segment each z-layer and then label in 3D.
        Default is False.
    gpu : bool, optional
        If True, use gpu for cellpose segmentation.
        Default: True
    model_type : str
        Cellpose model type ('cyto' or 'nuclei')
        Use 'cyto' for irregular nuclei.
        Default: 'cyto'.
    channels : tuple or list
        The 'channels' parameter of cellpose.
        Default: [0,0] (gray scale)
    diameter : int, optional
        Target cell diameter in pixels.
        If None, will be calculated as 12 microns converted to pixels.
        Default: None.
    remove_small_mode : str, optional
        '2D', or '3D'.
        Used to remove small cells/nuclei by volume (3D) or area (2D).
        For a thin stack (as in the example data), use '2D'.
        Default: 3D.
    remove_small_diam_fraction : float, optional
        Minimal diameter for the cells/nuclei.
        Provided as a fraction of the target diameters (the `diameter` parameter).
        Default: 0.5.
    clear_border : bool, optional
        If True, will remove cell touching image border (in xy only).
        Default: False
    add_to_input : bool
        If True, return an xarray dataset with combined input and output.
        Default: False
    return_cellpose_debug : bool
        If True, return flows, styles and diams together with masks.
        Default: False
    cellpose_kwargs : key value
        Cellpose arguments

    Returns
    -------
    masks = np.ndarray or xr.Dataset
        Segmented image or input with segmented image
    """
    if channels is None:
        channels = [0, 0]
    imgs, anisotropy = __get_images(dataset, do_3D, channel)
    imgs = [rescale_intensity(np.array(img)) for img in imgs]

    model = models.Cellpose(gpu=gpu, model_type=model_type)
    masks, flows, styles, diams = model.eval(imgs, anisotropy=anisotropy,
                                             diameter=diameter, channels=channels,
                                             **cellpose_kwargs)
    masks = __reshape_output(masks, dataset)

    if diameter is None:
        diameter = 12 / intake_io.get_spacing(dataset)[-1]
    masks = __combine_3D(masks, do_3D, diameter,
                         remove_small_mode=remove_small_mode,
                         remove_small_diam_fraction=remove_small_diam_fraction,
                         clear_border=clear_border)
    if add_to_input and not return_cellpose_debug:
        masks = __add_segmentation_to_image(dataset['image'].data, masks)
        if 'c' in dataset.dims:
            ch_names = list(dataset.coords['c'].data)
        else:
            ch_names = ['channel 0']
        masks = __image_to_dataset(masks,
                                   ch_names + ['Nuclei segmentation'],
                                   dataset)
    if return_cellpose_debug:
        return masks, flows, styles, diams
    else:
        return masks


def __add_segmentation_to_image(img, masks):
    if len(img.shape) > len(masks.shape):
        nshape = (img.shape[0] + 1,) + img.shape[1:]
    else:
        nshape = (2,) + img.shape
    new_img = np.zeros(nshape)
    new_img[:-1] = img
    new_img[-1] = masks
    return new_img.astype(np.uint16)


def __image_to_dataset(img, channel_names, template_dataset):
    coords = dict(c=channel_names)
    for c in ['x', 'y', 'z']:
        if c in template_dataset.dims:
            coords[c] = template_dataset.coords[c]
    dims = template_dataset['image'].dims
    if 'c' not in dims:
        dims = ('c',) + dims
    dataset = xr.Dataset(data_vars=dict(image=(dims, img)),
                         coords=coords,
                         attrs=template_dataset.attrs)
    return dataset


def segment_cells_batch(input_dir: str, output_dir: str, channel: int, **kwargs):
    """
    Segment all images in a give folder.

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Directory to save segmentation results.
        Segmentation is combined with the raw data into a multi-page tiff
    channel : int
        Channel number to use for segmentation, starting from 0.
    kwargs : key value
        Arguments for `segment_image` (see below)

    Attributes
    ---------
    do_3D : bool, optional
        If True, segment in the 3D mode with cellpose (computationally expensive).
        If False, segment each z-layer and then label in 3D.
        Default is False.
    gpu : bool, optional
        If True, use gpu for cellpose segmentation.
        Default: True
    model_type : str
        Cellpose model type ('cyto' or 'nuclei')
        Use 'cyto' for irregular nuclei.
        Default: 'cyto'.
    channels : tuple or list
        The 'channels' parameter of cellpose.
        Default: [0,0] (gray scale)
    diameter : int, optional
        Target cell diameter in pixels.
        If None, will be calculated as 12 microns converted to pixels.
        Default: None.
    remove_small_mode : str, optional
        '2D', or '3D'.
        Used to remove small cells/nuclei by volume (3D) or area (2D).
        For a thin stack (as in the example data), use '2D'.
        Default: 3D.
    remove_small_diam_fraction : float, optional
        Minimal diameter for the cells/nuclei.
        Provided as a fraction of the target diameter (the `diameter` parameter).
        Default: 0.5.
    cellpose_kwargs : key value
        Cellpose arguments

    """
    samples = walk_dir(input_dir)

    for i, sample in enumerate(samples):
        print(sample)
        print(fr'Processing sample {i + 1} of {len(samples)}')
        dataset = intake_io.imload(sample)
        output = segment_cells(dataset, channel, add_to_input=True, **kwargs)
        fn = sample[len(input_dir):].replace(os.path.splitext(sample)[-1], '.tif')
        os.makedirs(os.path.dirname(output_dir + fn), exist_ok=True)
        intake_io.imsave(output, output_dir + fn)


def __filter_laplace(img, minsize_um, maxsize_um, num_sigma, spacing):
    laplace = np.zeros(img.shape, dtype=np.float32)
    for sigma in np.linspace(minsize_um, maxsize_um, int(num_sigma), endpoint=True):
        gauss = filters.gaussian(img, sigma=sigma / spacing)
        laplace = np.max(np.stack([laplace, filters.laplace(gauss)]), axis=0)

    return laplace


def centers_to_markers(logblobs, img, bg_img, threshold_background):
    markers = np.zeros(img.shape)
    ind = np.int_(np.round_(logblobs[:, :len(img.shape)])).transpose()
    markers[tuple(ind)] = 1
    markers = markers * (img > bg_img * threshold_background)
    markers = ndimage.label(markers)[0]
    return markers


def calculate_background_image(img, cells, global_background=True,
                               global_background_percentile=95., background_percentile=50.):
    if cells is not None and len(np.unique(cells)) > 1:
        llist = np.unique(cells)[1:]
        if background_percentile == 50:
            bg = ndimage.median(img, cells, llist)
        else:
            bg = np.array([np.percentile(img[cells == lb], background_percentile)
                           for lb in llist])
        if global_background:
            bg_img = np.ones_like(img) * np.percentile(bg, global_background_percentile)
        else:
            bg_img = np.zeros_like(img)
            for i, l in enumerate(llist):
                bg_img[np.where(cells == l)] = bg[i]
    else:
        bg_img = np.zeros_like(img)
    return bg_img


def threshold_puncta(img, bg_img, cells, minsize_um, maxsize_um, num_sigma, spacing,
                     segmentation_mode, threshold_segmentation,
                     global_background=True, global_background_percentile=95., background_percentile=50.):
    if segmentation_mode == 0:
        intensity_image = __filter_laplace(img, minsize_um, maxsize_um, num_sigma, spacing)
        bg_img = np.ones_like(bg_img)
    elif segmentation_mode == 1:
        intensity_image = __filter_laplace(img, minsize_um, maxsize_um, num_sigma, spacing)
        bg_img = calculate_background_image(intensity_image, cells,
                                            global_background=global_background,
                                            global_background_percentile=global_background_percentile,
                                            background_percentile=background_percentile)
    elif segmentation_mode == 2:
        intensity_image = img
    else:
        raise ValueError(rf'{segmentation_mode} is invalid value for segmentation_mode, must be 0, 1, or 2')

    mask = intensity_image > threshold_segmentation * bg_img

    return mask


def segment_puncta(dataset, channel=None, cells=None, minsize_um=0.2, maxsize_um=2, num_sigma=5,
                   overlap=1, threshold_detection=0.001,
                   threshold_background=0, global_background=True,
                   global_background_percentile=95, background_percentile=50,
                   threshold_segmentation=50, segmentation_mode=1,
                   remove_out_of_cell=False, maxrad_um=None):
    """

    Parameters
    ----------
    dataset : xr.Dataset
        Image in the form of an xarray dataset (read with intake_io).
    channel : int, optional
        Channel number to use for segmentation, starting from 0.
        If the image has only one channel, this can be left out.
    cells : np.ndarray, optional
        Labeled segmentation masks for cells/nuclei.
        Default: None
    minsize_um : float
        Minimal sigma for the Laplacian of Gaussian detection (microns).
        Default: 0.2
    maxsize_um : float
        Maximal sigma for the Laplacian of Gaussian detection (microns).
        Default: 2
    num_sigma : int
        Number of sigma values for the Laplacian of Gaussian detection.
        Default: 5
    overlap : float
        Value between 0 and 1.
        If two blobs overlap by a fraction greater than this value,
                the smaller blob is eliminated.
        Default: 1 (blobs are removed only if overlapping completely)
    threshold_detection : float
        Threshold for detecting LoG blobs.
        The absolute lower bound for scale space maxima.
        Local maxima smaller than thresh are ignored.
        Reduce this to detect blobs with less intensities.
        Default: 0.001.
    threshold_background : float
        Threshold used to post-filter puncta in cells with diffuse signal.
        This threshold is provided relative to the median intensity inside cells.
        E.g, `threshold_background` = 2 will remove all puncta with intensity lower than two background values.
        Set to 0 to keep all puncta.
    global_background : bool
        If True, the background value is calculated globally as the `global_background_percentile` of all cells.
        Default: True
    global_background_percentile : float
        Percentile (between 0 and 100) of cell background values to calculate the global background value.
        Default: 95.
    background_percentile : float
        Percentile (between 0 and 100) of image intensity inside cell to calculate the background value.
        Default: 50 (median).
    threshold_segmentation : float
        Threshold for puncta segmentation.
        Used in combination with `segmentation_mode`.
        For segmentation_mode 0, choose values in the order of 0.001
        For segmentation_mode 1, choose values in the order of 50.
        For segmentation_mode 2, choose values in the order of 3.
        Reduce to detect more/larger puncta, increase to detect fewer/smaller puncta.
        Default: 50 (segmentation_mode 1).
    segmentation_mode : int
        0, 1, or 2.
        Determines the mode how `threshold_segmentation` is applied.
        0: apply absolute threshold in LoG space.
        1: apply threshold relative to background in LoG space.
        2: apply threshold relative to the background in image intensity space.
        Default: 1
    remove_out_of_cell : bool
        If True, remove all puncta (parts) that are not inside cells/nuclei.
        Default: False.
    maxrad_um : float
        If not None, remove puncta with a radius larger than this value.
        Default: None

    Returns
    -------
    puncta : np.ndarray
        Labeled segmentation mask for puncta
    """
    # get image and spacing
    spacing = np.array(intake_io.get_spacing(dataset))
    if 'c' in dataset.dims:
        img = dataset.loc[dict(c=dataset.coords['c'].data[channel])]['image'].data
    else:
        img = dataset['image'].data

    # find blob centers with scale-adapted LoG
    logblobs = blob_log(img,
                        min_sigma=minsize_um / spacing,
                        max_sigma=maxsize_um / spacing,
                        num_sigma=int(num_sigma),
                        overlap=overlap,
                        threshold=threshold_detection)

    # calculate background image
    bg_img = calculate_background_image(img, cells, global_background,
                                        global_background_percentile, background_percentile)

    # convert the blob centers to watershed markers, filter by background
    markers = centers_to_markers(logblobs, img, bg_img, threshold_background)

    # segment puncta
    mask = threshold_puncta(img, bg_img, cells, minsize_um, maxsize_um, num_sigma, spacing,
                            segmentation_mode, threshold_segmentation,
                            global_background, global_background_percentile, background_percentile)

    if remove_out_of_cell and cells is not None:
        mask = mask * (cells > 0)

    dist = ndimage.distance_transform_edt(mask, sampling=tuple(spacing))
    puncta = watershed(-dist, markers, mask=mask)
    if maxrad_um is not None:
        llist = np.unique(puncta)
        vol = ndimage.sum(puncta > 0, puncta, llist) * np.prod(spacing)
        if 'z' in dataset.dims:
            maxvol = 4. / 3 * np.pi * maxrad_um ** 3
        else:
            maxvol = np.pi * maxrad_um ** 2
        ix = np.in1d(puncta.ravel(), llist[vol > maxvol]).reshape(puncta.shape)
        puncta[ix] = 0

    return puncta


def segment_puncta_in_all_channels(dataset, puncta_channels=None, cell_segmentation=True,
                                   **puncta_kwargs):
    """
    Read input image and segment puncta in all specified channels.

    Parameters
    ----------
    dataset : xr.Dataset
        Image in the form of an xarray dataset (read with intake_io).
    puncta_channels : int or list of int
        (List of) puncta channel(s), starting from 0, to segment puncta in.
    cell_segmentation : bool
        If True, use the last channel of the input image as cell/nuclei mask.
        Default: True
    puncta_kwargs : key values
        Arguments for `segment_puncta` (see below).
        Each value may be provided as a single value or as a list of values for each puncta channel.

    Attributes
    ---------
    minsize_um : float
        Minimal sigma for the Laplacian of Gaussian detection (microns).
        Default: 0.2
    maxsize_um : float
        Maximal sigma for the Laplacian of Gaussian detection (microns).
        Default: 2
    num_sigma : int
        Number of sigma values for the Laplacian of Gaussian detection.
        Default: 5
    overlap : float
        Value between 0 and 1.
        If two blobs overlap by a fraction greater than this value,
                the smaller blob is eliminated.
        Default: 1 (blobs are removed only if overlapping completely)
    threshold_detection : float
        Threshold for detecting LoG blobs.
        The absolute lower bound for scale space maxima.
        Local maxima smaller than thresh are ignored.
        Reduce this to detect blobs with less intensities.
        Default: 0.001.
    threshold_background : float
        Threshold used to post-filter puncta in cells with diffuse signal.
        This threshold is provided relative to the median intensity inside cells.
        E.g, `threshold_background` = 2 will remove all puncta with intensity lower than two background values.
        Set to 0 to keep all puncta.
    global_background : bool
        If True, the background value is calculated globally as the `global_background_percentile` of all cells.
        Default: True
    global_background_percentile : float
        Percentile (between 0 and 100) of cell background values to calculate the global background value.
        Default: 95.
    background_percentile : float
        Percentile (between 0 and 100) of image intensity inside cell to calculate the background value.
        Default: 50 (median).
    threshold_segmentation : float
        Threshold for puncta segmentation.
        Used in combination with `segmentation_mode`.
        For segmentation_mode 0, choose values in the order of 0.001
        For segmentation_mode 1, choose values in the order of 50.
        For segmentation_mode 2, choose values in the order of 3.
        Reduce to detect more/larger puncta, increase to detect fewer/smaller puncta.
        Default: 50 (segmentation_mode 1).
    segmentation_mode : int
        0, 1, or 2.
        Determines the mode how `threshold_segmentation` is applied.
        0: apply absolute threshold in LoG space.
        1: apply threshold relative to background in LoG space.
        2: apply threshold relative to the background in image intensity space.
        Default: 1
    remove_out_of_cell : bool
        If True, remove all puncta (parts) that are not inside cells/nuclei.
        Default: False.
    maxrad_um : float
        If not None, remove puncta with a radius larger than this value.
        Default: None

    Returns
    ------
    output : xr.Dataset
        Image with added puncta segmentations

    """
    if 'c' in dataset.dims:
        puncta_channels = np.ravel(puncta_channels)
        ch_names = dataset.coords['c'].data
        if len(ch_names) <= np.max(puncta_channels) + 1:
            cell_segmentation = False
    else:
        cell_segmentation = False
        ch_names = ['ch0']
        puncta_channels = [0]
    if cell_segmentation:
        cells = dataset.loc[dict(c=ch_names[-1])]['image'].data
    else:
        cells = None
    for key in puncta_kwargs:
        param = np.ravel(puncta_kwargs[key])
        if not len(param) == len(puncta_channels):
            param = [param[0]] * len(puncta_channels)
        puncta_kwargs[key] = param

    output = dataset['image'].data
    for i, channel in enumerate(puncta_channels):
        cur_kwargs = dict()
        for key in puncta_kwargs:
            cur_kwargs[key] = puncta_kwargs[key][i]
        puncta = segment_puncta(dataset, cells=cells, channel=channel, **cur_kwargs)
        output = __add_segmentation_to_image(output, puncta)
    output = __image_to_dataset(output, list(ch_names) +
                                [rf'{cn} puncta' for cn in puncta_channels], dataset)
    return output


def __segment_puncta_in_all_channels(item, **kwargs):
    fn_in, fn_out = item
    dataset = intake_io.imload(fn_in)
    output = segment_puncta_in_all_channels(dataset=dataset, **kwargs)
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    intake_io.imsave(output, fn_out)


def segment_puncta_batch(input_dir: str, output_dir: str,
                         parallel: bool = True, n_jobs: int = 8,
                         **kwargs):
    """
    Segment puncta in all images in the input directory.

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory
    parallel : bool, optional
        If True, run the conversion in parallel.
        Default: True
    n_jobs : int, optional
        Number of jobs to run in parallel if `parallel` is True
        Default: 8
    kwargs : key value.
        Arguments for `segment_puncta_in_all_channels` (see below).

    Attributes
    ---------
    puncta_channels : int or list of int
        (List of) puncta channel(s), starting from 0, to segment puncta in.
    cell_segmentation : bool
        If True, use the last channel of the input image as cell/nuclei mask.
        Default: True
    minsize_um : float
        Minimal sigma for the Laplacian of Gaussian detection (microns).
        Default: 0.2
    maxsize_um : float
        Maximal sigma for the Laplacian of Gaussian detection (microns).
        Default: 2
    num_sigma : int
        Number of sigma values for the Laplacian of Gaussian detection.
        Default: 5
    overlap : float
        Value between 0 and 1.
        If two blobs overlap by a fraction greater than this value,
                the smaller blob is eliminated.
        Default: 1 (blobs are removed only if overlapping completely)
    threshold_detection : float
        Threshold for detecting LoG blobs.
        The absolute lower bound for scale space maxima.
        Local maxima smaller than thresh are ignored.
        Reduce this to detect blobs with less intensities.
        Default: 0.001.
    threshold_background : float
        Threshold used to post-filter puncta in cells with diffuse signal.
        This threshold is provided relative to the median intensity inside cells.
        E.g, `threshold_background` = 2 will remove all puncta with intensity lower than two background values.
        Set to 0 to keep all puncta.
    global_background : bool
        If True, the background value is calculated globally as the `global_background_percentile` of all cells.
        Default: True
    global_background_percentile : float
        Percentile (between 0 and 100) of cell background values to calculate the global background value.
        Default: 95.
    background_percentile : float
        Percentile (between 0 and 100) of image intensity inside cell to calculate the background value.
        Default: 50 (median).
    threshold_segmentation : float
        Threshold for puncta segmentation.
        Used in combination with `segmentation_mode`.
        For segmentation_mode 0, choose values in the order of 0.001
        For segmentation_mode 1, choose values in the order of 50.
        For segmentation_mode 2, choose values in the order of 3.
        Reduce to detect more/larger puncta, increase to detect fewer/smaller puncta.
        Default: 50 (segmentation_mode 1).
    segmentation_mode : int
        0, 1, or 2.
        Determines the mode how `threshold_segmentation` is applied.
        0: apply absolute threshold in LoG space.
        1: apply threshold relative to background in LoG space.
        2: apply threshold relative to the background in image intensity space.
        Default: 1
    remove_out_of_cell : bool
        If True, remove all puncta (parts) that are not inside cells/nuclei.
        Default: False.
    maxrad_um : float
        If not None, remove puncta with a radius larger than this value.
        Default: None

    """
    files = walk_dir(input_dir)
    items = [(fn, fn.replace(input_dir, output_dir)) for fn in files]

    if parallel:
        run_parallel(items=items, process=__segment_puncta_in_all_channels, max_threads=n_jobs, **kwargs)
    else:
        for item in tqdm(items):
            __segment_puncta_in_all_channels(item=item, **kwargs)


def substract_nuclei_from_cells(nuclei, cells, match_labels=True):
    # relable nuclei
    if match_labels:
        nuclei = (nuclei > 0) * cells

    # extract cytoplasm
    cytoplasm = cells * (nuclei == 0)
    return cytoplasm
