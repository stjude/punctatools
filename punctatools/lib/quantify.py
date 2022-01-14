import os

import intake_io
import numpy as np
import pandas as pd
from am_utils.parallel import run_parallel
from am_utils.utils import walk_dir, combine_statistics
from scipy import ndimage
from scipy.stats import entropy
from skimage.measure import regionprops_table
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


def __get_data(dataset, channel_names, puncta_channels):
    imgs = dataset['image'].data[:len(channel_names)]
    if len(dataset['image'].data) >= len(channel_names) + len(puncta_channels):
        puncta = dataset['image'].data[-len(puncta_channels):]
    else:
        raise ValueError(rf'No puncta segmentation found')
    if len(dataset['image'].data) >= len(channel_names) + len(puncta_channels) + 1:
        cells = dataset['image'].data[-len(puncta_channels) - 1]
    else:
        cells = np.zeros_like(imgs[-1])
    return imgs, cells, puncta


def __compute_volume_and_position(labels, spacing, name, img=None, img_name=''):
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


def __add_entropy_stats(stats, channel_data, ind, cur_cell_pix, channel_name):
    stats.at[ind, channel_name + ' entropy'] = entropy(np.histogram(channel_data[cur_cell_pix],
                                                                    bins=channel_data.max())[0])
    return stats


def __add_correlation_stats(stats, ind, channel_data1, channel_data2, cur_cell_pix, channel_names):
    mi = mutual_information_2d(channel_data1[cur_cell_pix],
                               channel_data2[cur_cell_pix],
                               bins=max([channel_data1[cur_cell_pix].max(),
                                         channel_data2[cur_cell_pix].max()]))
    corr = np.corrcoef(channel_data1[cur_cell_pix] * 1., channel_data2[cur_cell_pix] * 1.)[0, 1]

    stats.at[ind, 'Mutual information ' + channel_names[0] + ' vs ' + channel_names[1]] = mi
    stats.at[ind, 'Pearson correlation ' + channel_names[0] + ' vs ' + channel_names[1]] = corr
    return stats


def __add_cell_label(stats, cells):
    if 'z' in stats.columns:
        coords = np.int_(np.round_(stats[['z', 'y', 'x']].values))
    else:
        coords = np.int_(np.round_(stats[['y', 'x']].values))
    stats['cell_label'] = cells[tuple(coords.transpose())]
    return stats


def __summarize_puncta_stats(cell_stats, puncta_stats, puncta_channel):
    for i in range(len(cell_stats)):
        current_cell = puncta_stats[puncta_stats['cell_label'] == cell_stats['cell label'].iloc[i]]
        cell_stats.at[i, rf'number of {puncta_channel} puncta'] = len(current_cell)

        for col in ['puncta volume um', 'puncta volume pix', 'distance to nucleus border um']:
            colname = rf"average {puncta_channel} puncta {col} per nucleus"
            if len(current_cell) > 0:
                cell_stats.at[i, colname] = np.mean(current_cell[col])
            else:
                cell_stats.at[i, colname] = 0

        for col in ['puncta volume um', 'puncta volume pix']:
            colname = rf"total {puncta_channel} puncta {col} per nucleus"
            if len(current_cell) > 0:
                cell_stats.at[i, colname] = np.sum(current_cell[col])
            else:
                cell_stats.at[i, colname] = 0
    return cell_stats


def __total_intensities_in_out_puncta_per_cell(cell_stats, cells, puncta, puncta_channel, channel_data, channel):
    for label_img, location in zip([cells * (puncta > 0), cells * (puncta == 0)],
                                   [rf'inside {puncta_channel} puncta', rf'outside {puncta_channel} puncta']):
        intensity_stats = regionprops_table(label_image=label_img,
                                            intensity_image=channel_data,
                                            properties=['label', 'area', 'mean_intensity'])
        ind = cell_stats[cell_stats['cell label'].isin(intensity_stats['label'])].index

        cell_stats.at[ind, channel + ' mean intensity ' + location] = intensity_stats['mean_intensity']
        cell_stats.at[ind, channel + ' integrated intensity ' +
                      location] = np.int_(intensity_stats['mean_intensity'] * intensity_stats['area'])
    return cell_stats


def quantify(dataset, channel_names, puncta_channels):
    """
    Quantify cells and puncta in a segmented dataset.

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
    cell_stats : pd.DataFrame
        Statistics per individual cell/nucleus.
    puncta_stats : pd.DataFrame
        Statistics per individual punctum.
    """
    spacing = intake_io.get_spacing(dataset)
    if channel_names is None:
        channel_names = [rf"ch{i}" for i in range(len(dataset['c'].data) - len(puncta_channels) - 1)]
    puncta_channels = np.array(channel_names)[puncta_channels]
    channel_names = np.array(channel_names)
    imgs, cells, puncta = __get_data(dataset, channel_names, puncta_channels)

    # compute cell volume and positions
    cell_stats = __compute_volume_and_position(cells, spacing, 'cell')

    # compute intensities of all channels per cell
    for i in range(len(channel_names)):
        cell_stats = __add_intensity_stats(cell_stats, imgs[i], cells, channel_names[i], 'cell')

    # compute entropy and correlations of all channels per cell
    for ind in range(len(cell_stats)):
        cur_cell_pix = np.where(cells == cell_stats['cell label'].iloc[ind])
        for i in range(len(channel_names)):
            cell_stats = __add_entropy_stats(cell_stats, imgs[i], ind, cur_cell_pix, channel_names[i])

            for j in range(i + 1, len(channel_names)):
                cell_stats = __add_correlation_stats(cell_stats, ind, imgs[i], imgs[j], cur_cell_pix,
                                                     [channel_names[i], channel_names[j]])

    # quantify puncta
    dist_to_border = ndimage.morphology.distance_transform_edt(cells > 0, sampling=spacing)
    puncta_stats_all = pd.DataFrame()
    for p_i in range(len(puncta_channels)):
        # compute volume and positions of puncta
        puncta_stats = __compute_volume_and_position(puncta[p_i], spacing, 'puncta',
                                                     img=dist_to_border, img_name='distance to nucleus border um')
        puncta_stats = __add_cell_label(puncta_stats, cells)

        # compute intensities of all channels per puncta
        for i in range(len(channel_names)):
            puncta_stats = __add_intensity_stats(puncta_stats, imgs[i], puncta[p_i],
                                                 channel_names[i], 'puncta', bg_intensity=False)

        # summarize puncta stats
        cell_stats = __summarize_puncta_stats(cell_stats, puncta_stats, puncta_channels[p_i])

        # intensity stats per cell inside/outside puncta
        for i in range(len(channel_names)):
            cell_stats = __total_intensities_in_out_puncta_per_cell(cell_stats, cells, puncta[p_i],
                                                                    puncta_channels[p_i], imgs[i],
                                                                    channel_names[i])
        # combine puncta stats from all channels
        puncta_stats['channel'] = puncta_channels[p_i]
        puncta_stats_all = pd.concat([puncta_stats_all, puncta_stats], ignore_index=True)

    return cell_stats, puncta_stats_all


def __set_sample_name(stats, imgname):
    stats['Image name'] = imgname
    stats['sample'] = imgname.split('/')[-1]
    if len(imgname.split('/')) > 1:
        stats['condition'] = imgname.split('/')[-2]
    return stats


def __quantify(item, **kwargs):
    fn_in, fn_out_cells, fn_out_puncta, imgname = item
    dataset = intake_io.imload(fn_in)
    cell_stats, puncta_stats = quantify(dataset=dataset, **kwargs)

    cell_stats = __set_sample_name(cell_stats, imgname)
    puncta_stats = __set_sample_name(puncta_stats, imgname)

    os.makedirs(os.path.dirname(fn_out_cells), exist_ok=True)
    os.makedirs(os.path.dirname(fn_out_puncta), exist_ok=True)

    cell_stats.to_csv(fn_out_cells, index=False)
    puncta_stats.to_csv(fn_out_puncta, index=False)


def quantify_batch(input_dir: str, output_dir_cells: str, output_dir_puncta: str,
                   parallel: bool = True, n_jobs: int = 8,
                   **kwargs):
    """
    Quantify cells and puncta in all images in the input directory.

    input_dir : str
        Input directory
    output_dir_cells : str
        Output directory to save cell stats.
    output_dir_puncta : str
        Output directory to save puncta stats.
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
    items = [(fn, fn.replace(input_dir, output_dir_cells).replace('.tif', '.csv'),
              fn.replace(input_dir, output_dir_puncta).replace('.tif', '.csv'),
              fn[len(input_dir) + 1:])
             for fn in files]

    if parallel:
        run_parallel(items=items, process=__quantify, max_threads=n_jobs, **kwargs)
    else:
        for item in tqdm(items):
            __quantify(item=item, **kwargs)
    combine_statistics(output_dir_cells.rstrip('/') + '/')
    combine_statistics(output_dir_puncta.rstrip('/') + '/')
