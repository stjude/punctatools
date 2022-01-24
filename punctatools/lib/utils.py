import json
import os

import holoviews as hv
import intake_io
import numpy as np
import pandas as pd
import pylab as plt
from cellpose import plot
from skimage import io


def save_parameters(new_params, parameter_file):
    if not parameter_file.endswith('.json'):
        parameter_file += '.json'
    if os.path.exists(parameter_file):
        with open(parameter_file) as f:
            params = json.load(f)
        for key in new_params:
            params[key] = new_params[key]
    else:
        params = new_params

    with open(parameter_file, 'w') as f:
        json.dump(params, f, indent=4)
    return params


def load_parameters(variables, param_keys, param_matches):
    kwargs = dict()

    if 'parameter_file' in variables:
        with open(variables['parameter_file']) as f:
            params = json.load(f)
        for key in param_keys:
            if key in params.keys():
                kwargs[key] = params[key]
            else:
                raise ValueError(rf"Parameter {key} is not in the parameter list!")
        for key in param_matches:
            if param_matches[key] in params.keys():
                kwargs[key] = params[param_matches[key]]
            else:
                raise ValueError(rf"Parameter {key} is not in the parameter list!")

    else:
        for key in param_keys:
            if key in variables:
                kwargs[key] = variables[key]
            else:
                raise ValueError(rf"Parameter {key} value was not provided")
        for key in param_matches:
            if key in variables:
                kwargs[key] = variables[key]
            else:
                raise ValueError(rf"Parameter {key} value was not provided")

    return kwargs


def show_image_and_nuclei(ds, channels, nuclei=None, figsize=None):
    if figsize is None:
        figsize = 5
    chnames = ds['c'].data
    imgs = [ds.loc[dict(c=chnames[channel])]['image'] for channel in channels]
    if 'z' in ds.dims:
        imgs = [img.max('z') for img in imgs]

    cols = len(imgs)
    if nuclei is not None:
        cols += 1
    if cols > 1:
        fig, axs = plt.subplots(1, cols, figsize=(figsize * cols, figsize))
        for i in range(len(imgs)):
            plt.sca(axs[i])
            plt.title(rf"puncta channel {i}")
            io.imshow(imgs[i].data)
        if nuclei is not None:
            if len(nuclei.shape) > 2:
                nuclei = nuclei.max(0)
            plt.sca(axs[-1])
            plt.title("nuclei")
            io.imshow(nuclei)
        plt.show()
    else:
        plt.figure(figsize=(figsize, figsize))
        io.imshow(imgs[0].data)
        plt.show()


def show_image_and_segmentation(ds, channels, segmentations,
                                wh=400, cmap='viridis', holoviews=False):
    chnames = ds['c'].data
    imgs = [ds.loc[dict(c=chnames[channel])]['image'] for channel in channels]
    if 'z' in ds.dims:
        imgs = [img.max('z') for img in imgs]

    if holoviews:
        figure = None
        for i in range(len(imgs)):
            image = hv.Image(imgs[i], kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap)
            if figure is None:
                figure = image.opts(title=rf"puncta channel {i}")
            else:
                figure += image.opts(title=rf"puncta channel {i}")
            segm = segmentations[i]
            if len(segm.shape) > 2:
                segm = segm.max(0)
            segm_ds = imgs[i].copy()
            segm_ds.data = segm * imgs[i].data.max() / segm.max()
            figure += hv.Image(segm_ds, kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap).opts(
                title=rf"puncta channel {i}, segmentation")

        return figure
    else:
        cols = len(imgs) * 2
        fig, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))
        for i in range(len(imgs)):
            plt.sca(axs[i * 2])
            plt.title(rf"puncta channel {i}")
            io.imshow(imgs[i].data)

            plt.sca(axs[2 * i + 1])
            plt.title(rf"puncta channel {i}, segmentation")
            segm = segmentations[i]
            if len(segm.shape) > 2:
                segm = segm.max(0)

            io.imshow(segm)


def display_blobs(ds, channels, logblobs, wh=400, cmap='viridis', holoviews=True):
    chnames = ds['c'].data
    imgs = [ds.loc[dict(c=chnames[channel])]['image'] for channel in channels]
    if 'z' in ds.dims:
        imgs = [img.max('z') for img in imgs]
    spacing = intake_io.get_spacing(ds)

    if holoviews:
        figure = None
        for i in range(len(imgs)):
            image = hv.Image(imgs[i], kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap)
            if figure is None:
                figure = image.opts(title=rf"puncta channel {i}")
            else:
                figure += image.opts(title=rf"puncta channel {i}")
            lgblobs = logblobs[i]
            if len(lgblobs) > 0:
                lgblobs = lgblobs[:, -2:].copy()
                blobs = pd.DataFrame(lgblobs, columns=['y', 'x'])
                blobs['x'] = blobs['x'] * spacing[-1]
                blobs['y'] = blobs['y'] * spacing[-2]

                pts = hv.Points(blobs, kdims=['x', 'y']).opts(width=wh, height=wh,
                                                              line_alpha=0.7, fill_alpha=0.0,
                                                              color='red', size=10)
                figure += (image * pts).opts(title=rf"puncta channel {i}, blobs")
            else:
                figure += image.opts(title=rf"puncta channel {i}, blobs")

        return figure
    else:
        cols = len(imgs) * 2
        fig, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))
        for i in range(len(imgs)):
            plt.sca(axs[i * 2])
            plt.title(rf"puncta channel {i}")
            io.imshow(imgs[i].data)
            lgblobs = logblobs[i]

            plt.sca(axs[2 * i + 1])
            plt.title(rf"puncta channel {i}, blobs")
            io.imshow(imgs[i].data)
            if len(lgblobs) > 0:
                lgblobs = lgblobs[:, -2:].copy()
                plt.sca(axs[2 * i + 1])
                plt.scatter(lgblobs[:, 1], lgblobs[:, 0], edgecolors='red', facecolors='none', s=40)


def params_to_list(nchannels, *params):
    params = list(params)
    for i in range(len(params)):
        param = np.ravel(params[i])
        if not len(param) == nchannels:
            param = [param[0]] * nchannels
        params[i] = param
    return params


def get_value_from_list(channel, *params):
    params = list(params)
    for i in range(len(params)):
        params[i] = params[i][channel]
    return params


def convert_params(nchannels, channel, *params):
    params = params_to_list(nchannels, *params)
    params = get_value_from_list(channel, *params)
    return params


def display_roi_segmentation_results(masks, flows, dataset, channel, chnames, nimg=5):
    if 'c' in dataset.dims:
        imgs = dataset.loc[dict(c=chnames[channel])]['image'].data
    else:
        imgs = dataset['image'].data

    if 'z' not in dataset.dims:
        imgs = [imgs]
        masks = [masks]

    if len(imgs) > nimg:
        ind0 = int(len(imgs) / 2)
        ind = np.arange(ind0 - int(nimg / 2), ind0 + (nimg - int(nimg / 2)))
    else:
        ind = np.arange(len(imgs))

    for i in ind:
        maski = masks[i]
        flowi = flows[i][0]
        img = imgs[i]
        fig = plt.figure(figsize=(30, 10))
        if len(img.shape) > 2:
            img = img[int(img.shape[0] / 2)]
            maski = maski[int(maski.shape[0] / 2)]
            flowi = flowi[int(flowi.shape[0] / 2)]
        plot.show_segmentation(fig, img, maski, flowi, channels=[0, 0])
        plt.tight_layout()
        plt.show()


def crop_dataset(dataset, x, y, z, width, height, depth):
    sp = intake_io.get_spacing(dataset)[-1]

    ds_crop = dataset.copy()

    if x is not None and width is not None:
        ds_crop = ds_crop.loc[dict(x=slice(x * sp, (x + width - 1) * sp))]
    if y is not None and height is not None:
        ds_crop = ds_crop.loc[dict(y=slice(y * sp, (y + height - 1) * sp))]

    if 'z' in dataset.dims and z is not None and depth is not None:
        sp = intake_io.get_spacing(dataset)[0]
        ds_crop = ds_crop.loc[dict(z=slice(z * sp, (z + depth - 1) * sp))]

    sp = intake_io.get_spacing(dataset)[-1]
    ds_crop.coords['x'] = np.arange(ds_crop['image'].shape[-1]) * sp
    ds_crop.coords['y'] = np.arange(ds_crop['image'].shape[-2]) * sp
    return ds_crop
