# Image analysis of fluorescently tagged biomolecular condensates

## Installation

**Option 1:**

1. Download the latest version of the package
2. cd into the package directory
3. Install the package by running `pip install .`
    
You can install the punctatools into your base python environment, but we recommend creating 
a new [anaconda](https://docs.anaconda.com/anaconda/install/) 
or [mamba](https://github.com/mamba-org/mamba) environment. 

This installation should work with python versions 3.7-3.9.

**Option 2:**

Create a new conda environment with all dependencies from the provided yml file: 
   
`conda env create -f punctatools.yml`

To overwrite an old environment with the same name, run the above command with the `--force` flag:

`conda env create -f punctatools.yml --force`


## Usage

### 1. Prepare your images

- If your dataset is organized in a way that all channels and z-layers belonging to the same image 
  are combined in one file (e.g. multi-page tiff, or raw microscopy format), and one file corresponds
  to one image, you may skip this step and go directly to step 2. 
  Make sure that the pixel size and z-spacing are included in the metadata 
  of the image files: this information will later be used during 
  quantification to determine volumes of cells and puncta.
  
- If your dataset contains multiple positions or time points per file 
  (e.g., the Slidebook format), split the file into individual positions, e.g. using the provided 
  Fiji macro: [scripts/export_multipage.ijm](scripts/export_multipage.ijm). 
  Open the macro in [Fiji](https://imagej.net/software/fiji/) and run it. 
  When prompted, select the input directory with the images to convert, 
  and the output directory to save the results (the output directory will need to be 
  created beforehand).
  
- If you have channels and/or z-layers as individual files, combine the images to stacks
  with the provided conversion notebooks:

    1. Set up the parameter of the conversion with the 
    [setup_images_to_stack.ipynb](notebooks/setup_images_to_stack.ipynb) notebook. 
    Follow the instructions in the notebook.
    2. Convert the entire dataset.
       - Option 1: Run the [run_images_to_stack.ipynb](notebooks/run_images_to_stack.ipynb) 
         notebook.
       - Option 2: Run the [run_images_to_stack.py](scripts/run_images_to_stack.py) script: 
    
            ``python run_images_to_stack.py -p <parameter_file>``
    
            where `<parameter_file>` is the json file with parameters generated after running the set up 
    notebook ([setup_images_to_stack.ipynb](notebooks/setup_images_to_stack.ipynb)), e.g.:
         
            ``python run_images_to_stack.py -p parameters.json``
    
### 2. Segment cells or cell nuclei

If you don't have cell/nuclei stain and/or not interested in calculating the puncta statistics per cell, 
go to step 3.

1. Set up the parameter of the cell segmentation with the 
    [setup_cell_segmentation.ipynb](notebooks/setup_cell_segmentation.ipynb) notebook. 
    Follow the instructions in the notebook.
2. Segment the entire dataset.
   - Option 1: Run the [run_cell_segmentation.ipynb](notebooks/run_cell_segmentation.ipynb) 
     notebook.
   - Option 2: Run the [run_cell_segmentation.py](scripts/run_cell_segmentation.py) script: 

        ``python run_cell_segmentation.py -p <parameter_file>``

        where `<parameter_file>` is the json file with parameters generated after running the set up 
notebook ([setup_cell_segmentation.ipynb](notebooks/setup_cell_segmentation.ipynb)), e.g.:
     
        ``python run_cell_segmentation.py -p parameters.json``



### 3. Segment and quantify puncta

1. Set up the parameter of the puncta analysis with the 
    [setup_puncta_analysis.ipynb](notebooks/setup_puncta_analysis.ipynb) notebook. 
    Follow the instructions in the notebook.
2. Analyze the entire dataset.
   - Option 1: Run the [run_puncta_analysis.ipynb](notebooks/run_puncta_analysis.ipynb) 
     notebook.
   - Option 2: Run the [run_puncta_analysis.py](scripts/run_puncta_analysis.py) script: 

        ``python run_puncta_analysis.py -p <parameter_file>``

        where `<parameter_file>` is the json file with parameters generated after running the set up 
notebook ([setup_puncta_analysis.ipynb](notebooks/setup_puncta_analysis.ipynb)), e.g.:
     
        ``python run_puncta_analysis.py -p parameters.json``
     
