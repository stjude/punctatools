import json
import os

import holoviews as hv
import intake_io
import pandas as pd
import pylab as plt
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


def show_image_and_segmentation(ds, channel, segmentations,
                                wh=400, cmap='viridis', holoviews=False):
    chnames = ds['c'].data
    img = ds.loc[dict(c=chnames[channel])]['image']
    if 'z' in ds.dims:
        img = img.max('z')

    if holoviews:
        image = hv.Image(img, kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap)
        figure = image
        for i, segm in enumerate(segmentations):
            if len(segm.shape) > 2:
                segm = segm.max(0)
            segm_ds = img.copy()
            segm_ds.data = segm * img.data.max() / segm.max()
            figure += hv.Image(segm_ds, kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap)
        return figure
    else:
        if len(segmentations) > 0:
            cols = len(segmentations) + 1
            fig, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))
            plt.sca(axs[0])
            io.imshow(img.data)
            for i, segm in enumerate(segmentations):
                plt.sca(axs[i + 1])
                if len(segm.shape) > 2:
                    segm = segm.max(0)
                io.imshow(segm)
            plt.show()
        else:
            plt.figure(figsize=(7, 7))
            io.imshow(img.data)
            plt.show()


def display_blobs(ds, channel, logblobs, wh=400, cmap='viridis', holoviews=True):
    img = ds['image'].loc[dict(c=ds['c'].data[channel])]
    if 'z' in ds.dims:
        img = img.max('z')
    spacing = intake_io.get_spacing(ds)

    if holoviews:
        image = hv.Image(img, kdims=['x', 'y']).opts(width=wh, height=wh, cmap=cmap)
        figure = image
        for lgblobs in logblobs:
            if len(lgblobs) > 0:
                lgblobs = lgblobs[:, -2:].copy()
                blobs = pd.DataFrame(lgblobs, columns=['y', 'x'])
                blobs['x'] = blobs['x'] * spacing[-1]
                blobs['y'] = blobs['y'] * spacing[-2]

                pts = hv.Points(blobs, kdims=['x', 'y']).opts(width=wh, height=wh,
                                                              line_alpha=0.7, fill_alpha=0.0,
                                                              color='red', size=10)
                figure += image * pts
            else:
                figure += image

        return figure
    else:
        cols = len(logblobs) + 1
        fig, axs = plt.subplots(1, cols, figsize=(5 * cols, 5))
        plt.sca(axs[0])
        io.imshow(img.data)
        for i, lgblobs in enumerate(logblobs):
            lgblobs = lgblobs[:, -2:].copy()

            plt.sca(axs[i + 1])
            io.imshow(img.data)
            plt.sca(axs[i + 1])
            plt.scatter(lgblobs[:, 1], lgblobs[:, 0], edgecolors='red', facecolors='none', s=40)
