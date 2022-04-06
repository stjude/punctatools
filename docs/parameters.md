## List of parameters for ROI segmentation

Use the [setup_roi_segmentation.ipynb](../notebooks/setup_roi_segmentation.ipynb) notebook to adjust these parameters


| **Parameter**	| **Definition** |	**Recommended Initial Value** |	**Notes**
|-----|-----|-----|-----|
|input_dir|	Directory with images to be analyzed	| |	All z-layers and channels for a specific sample must be combined into a single file (see Step 3 of the Protocol Procedure)|
|output_dir |	Directory to save ROI segmentation results	| |	ROI masks will be added as an extra channel to the input image and saved in this directory|
|channel	| Channel index, starting from 0, that will be used to segment ROI	| |	Cellpose allows using nuclei channel to improve whole-cell segmentation. To use this option, provide two channel indices as a list, where the first index corresponds to the nuclei staining, and the second index corresponds to the cytoplasm staining. <br />  <br /> Examples: <br /> 0 – the first channel will be used to segment ROI (either cells or nuclei) <br /> [1, 0] – the second channel (1) will be used as an auxiliary nuclei stain, the first channel (0) will be used to segment whole cells |
|diameter |	Target ROI (cell or nucleus) diameter in pixels	| |	An example image displayed in the notebook will contain scale in pixels to help determine the target ROI diameter. Set to None to automatically detect the ROI diameter|
|model_type |	Cellpose model to use for segmentation: ‘nuclei’ for nucleus segmentation, ‘cyto’ or ‘cyto2’ for cell segmentation |	cyto	| We found that ‘cyto’ and ‘cyto2’ models work better than ‘nuclei’ for segmenting nuclei with irregular shapes|
|gpu |	If True, cellpose segmentation will run on GPU; if False, cellpose will use CPU |	True |	GPU processing is significantly faster; use gpu=True whenever possible|
|do_3D	| If True, cellpose segmentation is performed in 3D; if False, cellpose segments ROI in each individual z-layer, and the ROI are combined in 3D in the postprocessing |	False |	3D segmentation is resource intensive, though sometimes more accurate. If do_3D=True results in “CUDA out of memory” error, either set do_3D= False, or set gpu=False|
|flow_threshold |	Cellpose parameter: the maximum allowed error of the flows for each mask	| 0.4 |	Advanced parameter. Increase if cellpose returns too few masks; decrease if cellpose returns too many ill-shaped masks|
|cellprob_threshold |	Cellpose parameter: defines which pixels are used to run dynamics and determine masks |	0	| Advanced parameter. Decrease if cellpose returns too few ROI; increase if cellpose returns too many ROI; values should be between -6 and 6.|
|remove_ small_mode |	'2D', or '3D'. Used to remove small ROI by volume (3D) or area (2D) |	3D |	Set to ‘3D’ unless testing on cropped images. Set to ‘2D’ if the image contains only a few z-layers. If set to ‘3D’, small ROI are excluded based on volume; this will exclude a ROI if only small part of it is contained in the field of view. |
|remove_small_diam_fraction |	Size threshold used to exclude small ROI, provided as a fraction of the ‘diameter’ parameter |	0.5	| Advanced parameter. Increase to remove more ROI, decrease remove fewer ROI|
|parameter_ file |	File name used to save the parameter values |	parameters.json |	May include a complete path, or only a file name. If only a file name without path is provided, the file will be saved in the directory of the notebook|


## List of parameters for puncta segmentation and analysis 

Use the [setup_puncta_analysis.ipynb](../notebooks/setup_puncta_analysis.ipynb) notebook to adjust these parameters

| **Parameter**	| **Definition** |	**Recommended Initial Value** |	**Notes**
|-----|-----|-----|-----|
|parameter_file |	Parameter file with previously set up ROI segmentation parameters | | If ROI segmentation was performed, you can set this to the parameter file name used for ROI segmentation. Alternatively, set this to a new parameter file name and specify the following two parameters: “input_dir” and “cell_segmentation” |
|input_dir |	Directory with images to be analyzed | | All z-layers and channels for a specific sample must be combined into a single file (see Step 3 of the Protocol Procedure). If ROI segmentation was done, set this to the “output_dir” of the ROI segmentation. Alternatively, ignore this parameter and specify the “parameter_file”|
|roi_segmentation |	If True, the last channel of the input images will be used as ROI mask	| |	Set to False if the ROI segmentation step was skipped. Set to True if the image from “input_dir” contain ROI masks as the last channel. Alternatively, ignore this parameter and specify the “parameter_file”|
|output_dir	| Output directory to save puncta analysis results | | |		
|puncta_channels |	List of channel indices, starting form 0, that will be used to segment puncta | | Examples: <br /> [1] – puncta will be segmented in the second channel <br /> [2, 3] – puncta will be segmented in the third and fourth channels|
|minsize_um |	Minimum target puncta size in µm |	0.2 |	Will be used as the minimum sigma for the Laplacian of Gaussian detector. Decrease to detect smaller puncta, increase to avoid detection of smaller puncta|
|maxsize_um |	Maximum target puncta size in µm	| 2	| Will be used as the maximum sigma for the Laplacian of Gaussian detector. Increase to detect larger puncta, decrease to avoid detection of larger puncta|
|num_sigma |	Number of sigma values for the Laplacian of Gaussian detection |	5	| Advanced parameter. Decrease to save computational resources, increase to improve the accuracy of puncta centers detection|
|threshold_ detection |	Threshold used by LoG detector to exclude low intensity blobs | 	0.001	| Should be close to 0 and can be both positive and negative. Start with threshold_detection=0 and first adjust minsize_um and maxsize_um to make sure that all puncta of relevant size are detected. After that, gradually increase the value of threshold_detection to remove low-intensity detection|
|overlap |	Parameter used by the LoG detector to remove the smaller one of two overlapping blobs	| 1 |	Advanced parameter. Set to 1 to only remove completely overlapping blobs. Decrease to remove blobs that are further apart. Should be between 0 and 1|
|threshold_background	| Threshold used to remove low intensity puncta centers, provided relative to the ROI background value (see “background_percentile”) |	3 |	Example: <br />  threshold_ background=3 will remove all puncta centers with fluorescent intensity lower than 3 background values. <br /> Set to 0 to keep all puncta centers. <br /> Only applied if the ROI masks are provided|
|background_percentile	| Intensity percentile (between 0 and 100) used to calculate the background value of the ROI |	50	| Advanced parameter. 50 corresponds to the median value.|
|global_background |	If False, the background value is calculated individually for each ROI. If True, the background value is calculated globally as the global_background_percentile of all ROI |	False	| Set to False if there is a large range of cell fluorescence values.  This will increase sensitivity in cells with low fluorescence and decrease sensitivity in cells with high fluorescence|
|global_background_percentile |	Percentile (between 0 and 100) of ROI background values to calculate the global background value |	95 |	Advanced parameter. Only used if global_background=True|
|segmentation_mode	| Determines the way the “threshold_segmentation” is applied. For mode 0: absolute threshold is applied in LoG space; for mode 1: a threshold relative to the background is applied in LoG space; for mode 2: a threshold relative to the background is applied in image intensity space.	| 0	| Advanced parameter. Set to 0 if the background fluorescent signal in all ROI is relatively uniform. Set to 1 if there is a large range of ROI background fluorescence values|
|threshold_segmentation |	Threshold for puncta segmentation. Used in combination with the “segmentation_mode” |	0.001	| For mode 0, start with values between 0.001 and 0.003; for mode 1, start with values between 20 and 100; for mode 2, start with values between 2 and 3. <br /> Decrease or increase to detect more/bigger or fewer/smaller puncta|
|remove_out_of_roi	| If True, puncta (parts) that extend beyond ROI will be removed. If False, all puncta will be kept |	False	| |
|maxrad_um |	Maximum puncta radius in in µm. Used to remove large puncta |	None |	Set to None to keep all puncta
