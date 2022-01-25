import json
import os
import warnings

import intake_io
import numpy as np
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


def display_cellpose_results(masks, flows, dataset, channel, chnames, nimg=5):
    if 'c' in dataset.dims:
        imgs = dataset.loc[dict(c=chnames[channel])]['image'].data
    else:
        imgs = dataset['image'].data

    if 'z' not in dataset.dims:
        imgs = [imgs]
        masks = [masks]
        flows = [flows]

    if len(imgs) > nimg:
        ind0 = int(len(imgs) / 2)
        ind = np.arange(ind0 - int(nimg / 2), ind0 + (nimg - int(nimg / 2)))
    else:
        ind = np.arange(len(imgs))

    for i in ind:
        maski = masks[i]
        flowi = flows[i]
        img = imgs[i]
        fig = plt.figure(figsize=(30, 10))
        if len(img.shape) > 2:
            img = img[int(img.shape[0] / 2)]
            maski = maski[int(maski.shape[0] / 2)]
            flowi = flowi[int(flowi.shape[0] / 2)]
        plot.show_segmentation(fig, img, maski, flowi, channels=[0, 0])
        plt.tight_layout()
        plt.show()


def __get_data(ds, channels, channel_names, figsize):
    if figsize is None:
        figsize = 5
    if 'c' in ds.dims:
        chnames = ds['c'].data
        if channels is None:
            channels = np.arange(len(chnames))
        imgs = [ds.loc[dict(c=chnames[channel])]['image'] for channel in channels]
    else:
        chnames = ['channel 0']
        imgs = [ds['image']]
    if 'z' in ds.dims:
        imgs = [img.max('z') for img in imgs]
    if channel_names is None:
        channel_names = [chnames[channel] for channel in channels]
    imgs = [img.data for img in imgs]
    return imgs, channel_names, figsize


def show_dataset(ds, channels=None, channel_names=None, figsize=None):
    imgs, channel_names, figsize = __get_data(ds, channels, channel_names, figsize)
    show_imgs(imgs, channel_names, figsize)


def show_imgs(imgs, channel_names, figsize=None):
    if figsize is None:
        figsize = 5
    cols = len(imgs)
    if len(imgs[0].shape) > 2 and imgs[0].shape[-1] != 3:
        imgs = [img.max(0) for img in imgs]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if cols > 1:
            fig, axs = plt.subplots(1, cols, figsize=(figsize * cols, figsize))
            for i in range(len(imgs)):
                plt.sca(axs[i])
                plt.title(rf"{channel_names[i]}")
                io.imshow(imgs[i])
            plt.show()
        else:
            plt.figure(figsize=(figsize, figsize))
            io.imshow(imgs[0])
            plt.title(rf"{channel_names[0]}")
            plt.show()


def __plot_blobs(img, lgblobs, ax=None):
    io.imshow(img)
    if ax is not None:
        plt.sca(ax)

    if len(lgblobs) > 0:
        lgblobs = lgblobs[:, -2:].copy()
        plt.scatter(lgblobs[:, 1], lgblobs[:, 0], edgecolors='red', facecolors='none', s=40)


def display_blobs(ds, logblobs, channels=None, channel_names=None,
                  blobname='LoG blobs', figsize=None):
    imgs, channel_names, figsize = __get_data(ds, channels, channel_names, figsize)

    cols = len(imgs)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if cols > 1:
            fig, axs = plt.subplots(1, cols, figsize=(figsize * cols, figsize))
            for i in range(len(imgs)):
                plt.sca(axs[i])
                __plot_blobs(imgs[i], logblobs[i], axs[i])
                plt.sca(axs[i])
                plt.title(rf"{channel_names[i]}: {blobname}")
            plt.show()
        else:
            plt.figure(figsize=(figsize, figsize))
            __plot_blobs(imgs[0], logblobs[0])
            plt.title(rf"{channel_names[0]}: {blobname}")
            plt.show()
