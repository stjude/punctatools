import numpy as np

from skimage.feature import blob_log
from skimage.segmentation import watershed
from skimage import filters
from scipy import ndimage

EPS = np.finfo(float).eps

def rescale_intensity(x, quantiles=(0.0025, 0.9975)):
    mn, mx = [np.percentile(x, p*100) for p in quantiles]
    if mx > mn + 5:
        return np.clip((x.astype(np.float32) - mn) / (mx - mn), 0, 1)    
    else:
        return np.zeros(x.shape, dtype=np.float32)


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

    Returns
    -------
    mi: float
        the computed similariy measure

    """

    jh = np.histogram2d(x, y, bins=bins)[0]

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2)))

    return mi


def segment_puncta(img, cells, scale, minsize_um, maxsize_um, num_sigma, 
                   overlap, threshold_detection, threshold_background, 
                   threshold_segmentation, segmentation_mode=0,
                   return_blob_centers=False, **kwargs_to_ignore):
    
    # filter the puncta channel with scale-adapted LoG
    if segmentation_mode in [0,1]:
        laplace = []
        for sigma in np.linspace(minsize_um, maxsize_um, num_sigma, endpoint=True):
            gauss = filters.gaussian(img, sigma=sigma/scale)
            laplace.append(filters.laplace(gauss))
        laplace = np.array(laplace)
        laplace = laplace.max(0)
    
    # find blob centers with scale-adapted LoG
    logblobs = blob_log(img, 
                        min_sigma=minsize_um/scale, 
                        max_sigma=maxsize_um/scale, 
                        num_sigma=num_sigma, 
                        overlap=overlap, 
                        threshold=threshold_detection)
    
    # calculate background intensity as median puncta signal in each cell
    llist = np.unique(cells)[1:]
    bg = ndimage.median(img, cells, llist)
    bg_img = np.zeros_like(img)
    for i, l in enumerate(llist):
        bg_img[np.where(cells == l)] = bg[i]
    
    # convert the blob centers to watershed markers
    markers = np.zeros(img.shape)
    ind = np.int_(np.round_(logblobs[:, :3]))
    markers[ind[:,0], ind[:,1], ind[:,2]] = 1
    markers = markers * (img > bg_img * threshold_background)
    markers = ndimage.label(markers)[0]
    
    # segment puncta in the LoG-filtered image by thresholding and watershed
    if segmentation_mode == 0:
        mask = laplace > threshold_segmentation
    elif segmentation_mode == 1:     
        bg = ndimage.median(laplace, cells, llist)
        bg_img = np.zeros_like(laplace)
        for i, l in enumerate(llist):
            bg_img[np.where(cells == l)] = bg[i]
        mask = (laplace > threshold_segmentation * bg_img) * (cells > 0)
    elif segmentation_mode == 2:
        mask = (img > threshold_segmentation * bg_img) * (cells > 0)  
    else:
        raise ValueError(rf'{segmentation_mode} is invalid value for segmentation_mode, must be 0, 1, or 2')
    dist = ndimage.distance_transform_edt(mask, sampling=scale)
    puncta = watershed(-dist, markers, mask=mask)
    if return_blob_centers:
        return puncta, logblobs
    else:
        return puncta






