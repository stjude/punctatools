import os

import intake_io
import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel
from am_utils.utils import walk_dir, combine_statistics
from scipy import ndimage
from scipy.stats import entropy, pearsonr
from skimage.measure import regionprops_table, marching_cubes, mesh_surface_area
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

EPS = np.finfo(float).eps


def mutual_information_2d(x, y, bins=256):
    """
    Computes mutual information between two 1D variate from a
    joint histogram.

    Adapted from here: https://github.com/mutualinfo/mutual_info

    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    bins : int, optional
        Number of bins for the 2D histogram.
        Default: 256

    Returns
    -------
    mi: float
        The computed similarity measure

    """
    jh = np.histogram2d(x, y, bins=bins)[0]

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2)))

    return mi


def sphericity(area, volume):
    """
    Sphericity calculation as in https://en.wikipedia.org/wiki/Sphericity
    """
    return np.pi**(1./3) * (6 * volume)**(2./3) / area


def __get_data(dataset, channel_names, puncta_channels):
    imgs = dataset['image'].data[:len(channel_names)]
    if len(dataset['image'].data) >= len(channel_names) + len(puncta_channels):
        puncta = dataset['image'].data[-len(puncta_channels):]
    else:
        raise ValueError(rf'No puncta segmentation found')
    if len(dataset['image'].data) >= len(channel_names) + len(puncta_channels) + 1:
        roi = dataset['image'].data[-len(puncta_channels) - 1]
    else:
        roi = np.zeros_like(imgs[-1])
    return imgs, roi, puncta


def __compute_volume_shape_position(labels, spacing, name, img=None, img_name=''):
    properties = ['label', 'area', 'centroid']
    if img is not None:
        properties += ['mean_intensity']
    stats = pd.DataFrame(regionprops_table(label_image=labels, intensity_image=img,
                                           properties=properties))

    ncols = {'area': rf'{name} volume pix',
             'centroid-0': 'z',
             'centroid-1': 'y',
             'centroid-2': 'x',
             'label': rf'{name} label'}
    if len(labels.shape) == 2:
        ncols['centroid-0'] = 'y'
        ncols['centroid-1'] = 'x'
        ncols.pop('centroid-2')
    stats = stats.rename(columns=ncols)
    if img is not None:
        stats = stats.rename(columns={'mean_intensity': img_name})
    stats[rf'{name} volume um'] = stats[rf'{name} volume pix'] * np.prod(spacing)

    surf_area = [mesh_surface_area(*marching_cubes(labels == lb, level=None, spacing=spacing)[:2])
                 for lb in np.unique(labels)[1:]]
    stats['sphericity'] = sphericity(np.array(surf_area), stats[rf'{name} volume um'])
    return stats


def __add_intensity_stats(stats, channel_data, labels, channel_name, name, bg_intensity=True):
    intensity_stats = regionprops_table(label_image=labels,
                                        intensity_image=channel_data,
                                        properties=['label', 'mean_intensity'])
    stats[rf'{channel_name} mean intensity per {name}'] = intensity_stats['mean_intensity']
    stats[rf'{channel_name} integrated intensity per {name}'] = stats[rf'{channel_name} mean intensity per {name}'] \
                                                                * stats[rf'{name} volume pix']
    if bg_intensity:
        stats[channel_name + ' mean background intensity'] = np.mean(channel_data[np.where(labels == 0)])
        stats[channel_name + ' integrated background intensity'] = np.sum(channel_data[np.where(labels == 0)])
    return stats


def __add_entropy_stats(stats, channel_data, ind, cur_roi_pix, channel_name):
    stats.at[ind, channel_name + ' entropy'] = entropy(np.histogram(channel_data[cur_roi_pix],
                                                                    bins=channel_data.max())[0])
    return stats


def __add_correlation_stats(stats, ind, channel_data1, channel_data2, cur_cell_pix, channel_names):
    if len(channel_data1[cur_cell_pix]) >= 2:
        mi = mutual_information_2d(channel_data1[cur_cell_pix],
                                   channel_data2[cur_cell_pix],
                                   bins=max([channel_data1[cur_cell_pix].max(),
                                             channel_data2[cur_cell_pix].max()]))
        corr, pval = pearsonr(channel_data1[cur_cell_pix] * 1., channel_data2[cur_cell_pix] * 1.)

        stats.at[ind, 'Mutual information ' + channel_names[0] + ' vs ' + channel_names[1]] = mi
        stats.at[ind, 'Pearson correlation coefficient ' + channel_names[0] + ' vs ' + channel_names[1]] = corr
        stats.at[ind, 'Pearson correlation p value ' + channel_names[0] + ' vs ' + channel_names[1]] = pval
    return stats


def __add_coloc_stats(stats, ind, cur_roi_pix, overlap, union, chname):
    coloc = np.sum((overlap[cur_roi_pix] > 0) * 1) / np.sum(union[cur_roi_pix])

    stats.at[ind, 'Overlap coefficient ' + chname] = coloc
    return stats


def __add_roi_label(stats, roi):
    if 'z' in stats.columns:
        coords = np.int_(np.round_(stats[['z', 'y', 'x']].values))
    else:
        coords = np.int_(np.round_(stats[['y', 'x']].values))
    stats['ROI label'] = roi[tuple(coords.transpose())]
    return stats


def __summarize_puncta_quantifications(roi_quant, puncta_quant, puncta_channel):
    for i in range(len(roi_quant)):
        current_cell = puncta_quant[puncta_quant['ROI label'] == roi_quant['ROI label'].iloc[i]]
        roi_quant.at[i, rf'number of {puncta_channel} puncta'] = len(current_cell)

        for col in ['puncta volume um', 'puncta volume pix', 'distance to ROI border um', 'sphericity']:
            colname = rf"average {puncta_channel} puncta {col} per ROI"
            colname = colname.replace('puncta puncta', 'puncta')
            if len(current_cell) > 0:
                roi_quant.at[i, colname] = np.mean(current_cell[col])
            else:
                roi_quant.at[i, colname] = 0

        for col in ['puncta volume um', 'puncta volume pix']:
            colname = rf"total {puncta_channel} puncta {col} per ROI"
            colname = colname.replace('puncta puncta', 'puncta')
            if len(current_cell) > 0:
                roi_quant.at[i, colname] = np.sum(current_cell[col])
            else:
                roi_quant.at[i, colname] = 0
    return roi_quant


def __total_intensities_in_out_puncta_per_cell(roi_quant, roi, puncta, puncta_channel, channel_data, channel):
    for label_img, location in zip([roi * (puncta > 0), roi * (puncta == 0)],
                                   [rf'inside {puncta_channel} puncta', rf'outside {puncta_channel} puncta']):
        intensity_stats = regionprops_table(label_image=label_img,
                                            intensity_image=channel_data,
                                            properties=['label', 'area', 'mean_intensity'])
        ind = roi_quant[roi_quant['ROI label'].isin(intensity_stats['label'])].index

        roi_quant.at[ind, channel + ' mean intensity ' + location] = intensity_stats['mean_intensity']
        roi_quant.at[ind, channel + ' integrated intensity ' +
                     location] = np.int_(intensity_stats['mean_intensity'] * intensity_stats['area'])
    return roi_quant


def quantify(dataset, channel_names, puncta_channels):
    """
    Quantify ROI (cells/nuclei) and puncta in a segmented dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Image in the form of an xarray dataset (read with intake_io).
        Should include the original data, cell segmentation, and puncta segmentation.
    channel_names : list of str
        Names of the image channels.
    puncta_channels : list of int
        Indices of puncta channels, starting from 0.

    Returns
    -------
    roi_quant : pd.DataFrame
        Statistics per individual cell/nucleus.
    puncta_quant : pd.DataFrame
        Statistics per individual punctum.
    """
    spacing = intake_io.get_spacing(dataset)
    if channel_names is None:
        channel_names = [rf"ch{i}" for i in range(len(dataset['c'].data) - len(puncta_channels) - 1)]
    puncta_channels = np.array(channel_names)[puncta_channels]
    channel_names = np.array(channel_names)
    imgs, roi, puncta = __get_data(dataset, channel_names, puncta_channels)

    # compute cell volume and positions
    roi_quant = __compute_volume_shape_position(roi, spacing, 'ROI')

    # compute intensities of all channels per cell
    for i in range(len(channel_names)):
        roi_quant = __add_intensity_stats(roi_quant, imgs[i], roi, channel_names[i], 'ROI')

    # calculate colocalized puncta
    n = len(puncta_channels)
    p_union = []
    for pi1 in range(n):
        for pi2 in range(pi1 + 1, n):
            p_intersect = puncta[pi1].astype(np.int64) * puncta[pi2].astype(np.int64)
            p_intersect = relabel_sequential(p_intersect)[0]
            puncta = np.concatenate([puncta, np.expand_dims(p_intersect, 0)], axis=0)
            puncta_channels = np.concatenate([puncta_channels,
                                              np.array([rf"{puncta_channels[pi1]}_{puncta_channels[pi2]}_coloc"])])
            p_union.append(((puncta[pi1] + puncta[pi2]) > 0) * 1)

    # compute entropy, colocalization and correlations of all channels per cell
    for ind in range(len(roi_quant)):
        cur_roi_pix = np.where(roi == roi_quant['ROI label'].iloc[ind])
        for i in range(len(channel_names)):
            roi_quant = __add_entropy_stats(roi_quant, imgs[i], ind, cur_roi_pix, channel_names[i])

            for j in range(i + 1, len(channel_names)):
                roi_quant = __add_correlation_stats(roi_quant, ind, imgs[i], imgs[j], cur_roi_pix,
                                                    [channel_names[i], channel_names[j]])

        for i in range(len(p_union)):
            roi_quant = __add_coloc_stats(roi_quant, ind, cur_roi_pix,
                                          puncta[n + i], p_union[i], puncta_channels[n + i])

    # quantify puncta
    dist_to_border = ndimage.morphology.distance_transform_edt(roi > 0, sampling=spacing)
    puncta_quant_all = pd.DataFrame()
    for p_i in range(len(puncta_channels)):
        # compute volume and positions of puncta
        puncta_quant = __compute_volume_shape_position(puncta[p_i], spacing, 'puncta',
                                                       img=dist_to_border, img_name='distance to ROI border um')
        puncta_quant = __add_roi_label(puncta_quant, roi)

        # compute intensities of all channels per puncta
        for i in range(len(channel_names)):
            puncta_quant = __add_intensity_stats(puncta_quant, imgs[i], puncta[p_i],
                                                 channel_names[i], 'puncta', bg_intensity=False)

        # summarize puncta stats
        roi_quant = __summarize_puncta_quantifications(roi_quant, puncta_quant, puncta_channels[p_i])

        # intensity stats per cell inside/outside puncta
        for i in range(len(channel_names)):
            roi_quant = __total_intensities_in_out_puncta_per_cell(roi_quant, roi, puncta[p_i],
                                                                   puncta_channels[p_i], imgs[i],
                                                                   channel_names[i])

        # compute correlations of all channels per puncta
        for ind in range(len(puncta_quant)):
            cur_puncta_pix = np.where(puncta[p_i] == puncta_quant['puncta label'].iloc[ind])
            for i in range(len(channel_names)):
                for j in range(i + 1, len(channel_names)):
                    puncta_quant = __add_correlation_stats(puncta_quant, ind, imgs[i], imgs[j], cur_puncta_pix,
                                                           [channel_names[i], channel_names[j]])

        # combine puncta stats from all channels
        puncta_quant['channel'] = puncta_channels[p_i]
        puncta_quant_all = pd.concat([puncta_quant_all, puncta_quant], ignore_index=True)

    return roi_quant, puncta_quant_all


def __set_sample_name(stats, imgname):
    stats['Image name'] = imgname
    stats['sample'] = imgname.split('/')[-1]
    if len(imgname.split('/')) > 1:
        stats['condition'] = imgname.split('/')[-2]
    return stats


def __quantify(item, **kwargs):
    fn_in, fn_out_roi, fn_out_puncta, imgname = item
    dataset = intake_io.imload(fn_in)
    roi_quant, puncta_quant = quantify(dataset=dataset, **kwargs)

    roi_quant = __set_sample_name(roi_quant, imgname)
    puncta_quant = __set_sample_name(puncta_quant, imgname)

    os.makedirs(os.path.dirname(fn_out_roi), exist_ok=True)
    os.makedirs(os.path.dirname(fn_out_puncta), exist_ok=True)

    roi_quant.to_csv(fn_out_roi, index=False)
    puncta_quant.to_csv(fn_out_puncta, index=False)


def quantify_batch(input_dir: str, output_dir_roi: str, output_dir_puncta: str,
                   parallel: bool = True, n_jobs: int = 8,
                   **kwargs):
    """
    Quantify cells and puncta in all images in the input directory.

    input_dir : str
        Input directory
    output_dir_roi : str
        Output directory to save measurements individual ROI (cells or nuclei).
    output_dir_puncta : str
        Output directory to save measurements for individual puncta.
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
    channel_names : list of str
        Names of the image channels.
    puncta_channels : list of int
        Indices of puncta channels, starting from 0.
    """

    files = walk_dir(input_dir)
    items = [(fn, fn.replace(input_dir, output_dir_roi).replace('.tif', '.csv'),
              fn.replace(input_dir, output_dir_puncta).replace('.tif', '.csv'),
              fn[len(input_dir) + 1:])
             for fn in files]

    if parallel:
        run_parallel(items=items, process=__quantify, max_threads=n_jobs, **kwargs)
    else:
        for item in tqdm(items):
            __quantify(item=item, **kwargs)
    combine_statistics(output_dir_roi.rstrip('/') + '/')
    combine_statistics(output_dir_puncta.rstrip('/') + '/')
