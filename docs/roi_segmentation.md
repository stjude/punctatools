##	ROI segmentation:

ROI segmentation is done using [cellpose](https://github.com/mouseland/cellpose) 
and is done either in a 3D mode or 2D mode, depending on the user choice. 

### Cellpose segmentation

If the 3D mode is chosen, cellpose outputs 3D labeled masks of cells or nuclei. 

If the 2D mode is chosen, cellpose segments and labels cells / nuclei in 
individual z-layers. In this case, the labels for the same cell 
in different z-layers do not match and need to be relabeled to generate 
3D labeled masks, which is done as follows:

1.	Calculate the total foreground (cell) area in each z-layer.
2.	Detect the layer with the largest total cell area and extract the 
      cellpose labels from this layer; if the z-stack has more than 21 layers, 
      the first and the last 10 layers are not considered in this step.
3.	Convert all z-layer labels to binary masks (0 for the background, 1 for cells).
4.	Multiply each binary mask by the labels from step 2.

This relabeling procedure works well for samples where cells/nuclei are 
arranged in one layer and don’t lie on top of each other. 

This procedure combined with the 2D mode of cellpose is more memory efficient 
and should be preferred for such “thin” samples. 

The 3D mode of cellpose was more accurate in some of our tests, 
but it required more GPU memory, or significantly more computational time 
if the available GPU memory was not sufficient, and we had to use the CPU.

### Filtering of Cellpose masks

After obtaining a 3D labeled cell mask – either directly from cellpose or after 
relabeling – cells smaller than the minimum size are removed. 

The minimum size is specified as a fraction of the “diameter” parameter 
used by the cellpose, e.g., if the cellpose “diameter” is 120 pixels, 
and the fraction (“remove_small_diameter_fraction”) is 0.5, 
then all cells (or nuclei) with a diameter smaller than 60 pixels are removed. 

Technically, the removal is implemented by filtering the 3D labels by their 
volume, where the minimum volume is calculated from the minimum diameter 
using the sphere volume formula. 

In case of a very thin image (with only a few z-layers, 
as in the provided test examples), we provide an option of filtering 
by area rather than volume. 

For such thin images, the cell volume is not an adequate representation of 
cell size, since only a small fraction of the cell is contained in the image. 
In this case, we use minimum cell area (calculated as the circle area 
from the minimum diameter) as a criterion to filter cells based on 
their area averaged over all z-layers. 

The option to filter by volume or area is specified by 
the “remove_small_mode” parameters, which is set to either “3D” or “2D”.

Finally, we provide an option to remove cells at the image border, 
which is specified by the “clear_border” parameter (True or False). 
This option removes cells/nuclei that are touching the xy border of the image. 
The cell removal is not applied in z, since most of the cells touch 
image border in z due to the thin sample nature.
