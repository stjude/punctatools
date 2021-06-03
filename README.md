# Image analysis of NUP98 fusion protein puncta

## Analysis steps

1. Convert images with [images_to_stack.ipynb](puncta_analysis/images_to_stack.ipynb)
2. Segment cell nuclei with [segment_cells.ipynb](puncta_analysis/segment_cells.ipynb)
3. Adjuste puncta detection parameters with [adjust_detection_parameters.ipynb](puncta_analysis/adjust_detection_parameters.ipynb)
4. Segment and quantify puncta with [analyze_peaks_per_cell.ipynb](puncta_analysis/analyze_peaks_per_cell.ipynb)

## Installation

1. Install [anaconda](https://docs.anaconda.com/anaconda/install/)

2. Create a new conda environment from the provided yml file: `conda env create -f env/nup98_puncta.yml`
