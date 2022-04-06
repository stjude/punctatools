## Measurements for individual puncta 

| **Measurement** |	**Definition** |
|-----|-----|
Image name	| Source file name, including subdirectory
puncta label |	Unique puncta ID; matches the pixel value in the puncta segmentation mask
ROI label |	ID of the ROI (cell or nucleus) the punctum belongs to; 0 corresponds to puncta located outside ROI
channel |	Channel name in which the punctum was segmented
x |	the x-coordinate (pixels) of the punctum within the image
y |	the y-coordinate (pixels) of the punctum within the image
z |	the z-coordinate (pixels) of the punctum within the image
puncta volume pix |	Volume (or area, for 2D images) of the punctum in pixels.
puncta volume um |	Volume of the punctum in µm3 (or area in µm2, for 2D images)
distance to ROI border um |	Distance in µm (in 3D) from the punctum’s center of mass to the border of the cell or nucleus; 0 corresponds to puncta located in the background
[FL] mean intensity per puncta	| Average pixel intensity of the [Fl] channel inside the punctum. Calculated for each fluorescent channel in the image
[FL] integrated intensity per puncta |	The sum of all pixel intensities of the [Fl] channel inside the punctum. Calculated for each fluorescent channel in the image
Pearson correlation coefficient [Fl] vs [Fl*] |	Pearson correlation coefficient between each pair of fluorescent channels [Fl] and [Fl*] inside the punctum
Pearson correlation p value [Fl] vs [Fl*] |	p-value for the Pearson correlation coefficient between each pair of fluorescent channels [Fl] and [Fl*] inside the punctum
Mutual information [Fl] vs [Fl*] |	Mutual information between each pair of fluorescent channels [Fl] and [Fl*] inside the punctum. This is a measure of correlation between channels



## Measurements for individual ROIs (cells or nuclei)

| **Measurement** |	**Definition** |
|-----|-----|
Image name |	Source file name, including subdirectory
ROI label |	Unique ROI (cell or nucleus) ID; matches the pixel value in the ROI segmentation mask
x |	The x-coordinate (pixels) of the ROI within the image
y |	The y-coordinate (pixels) of the ROI within the image
z |	The z-coordinate (pixels) of the ROI within the image
ROI volume pix |	Volume (or area, for 2D images) of the ROI in pixels.
ROI volume um |	Volume of the ROI in µm3 (or area in µm2, for 2D images)
[Fl] mean intensity per ROI |	Average pixel intensity of the [Fl] channel inside the ROI. Calculated for each fluorescent channel in the image
[Fl] integrated intensity per ROI |	The sum of all pixel intensities of the [Fl] channel inside the ROI. Calculated for each fluorescent channel in the image
[Fl] mean background intensity |	Average pixel intensity of the [Fl] channel in the background (outsize of ROI); will have the same value for all ROI in the image
[Fl] integrated background intensity |	The sum of all pixel intensities of the [Fl] channel in the background (outsize of ROI); will have the same value for all ROI in the image
[Fl] entropy |	Entropy of the [Fl] channel inside the ROI
Pearson correlation coefficient [Fl] vs [Fl*]	| Pearson correlation coefficient between each pair of fluorescent channels [Fl] and [Fl*] inside the ROI
Pearson correlation p value [Fl] vs [Fl*] |	p-value for the Pearson correlation coefficient between each pair of fluorescent channels [Fl] and [Fl*] inside the ROI
Mutual information [Fl] vs [Fl*] |	Mutual information between each pair of fluorescent channels [Fl] and [Fl*] inside the ROI
number of [P] puncta |	Number of puncta detected in the [P] channel and assigned to the current ROI. This will correspond to the puncta measurements from puncta_stats.csv with matching “Image name” and “cell label” values and “channel”=[P]. Calculated for each channel [P] specified as puncta channel
average [P] puncta volume pix per ROI |	Average volume in pixels of puncta detected from channel [P] in the current ROI
average [P] puncta volume um per ROI |	Average volume in µm3 (or area in µm2, for 2D images) of puncta detected from channel [P] in the current ROI
total [P] puncta volume pix per ROI |	Total volume in pixels of puncta detected from channel [P] in the current ROI
total [P] puncta volume um per ROI |	Total volume in µm3 (or area in µm2, for 2D images) of puncta detected from channel [P] in the current ROI
average [P] puncta distance to ROI border um per nucleus |	Average distance in µm (in 3D) from the puncta’s centers of mass to the ROI border 
[Fl] mean intensity inside [P] puncta |	The average intensity of the [Fl] channel inside puncta detected in the [P] channel in the current ROI. For [Fl]=[P], this will correspond to the dense phase concentration
[Fl] mean intensity outside [P] puncta |	The average intensity of the [Fl] channel outside puncta detected in the [P] channel in the current ROI. For [Fl]=[P], this will correspond to the light phase concentration
[Fl] integrated intensity inside [P] puncta |	The sum of all intensities of the [Fl] channel inside puncta detected in the [P] channel in the current ROI.  
[FL] integrated intensity outside [P] puncta |	The sum of all intensities of the [Fl] channel outside puncta detected in the [P] channel in the current ROI.  
Overlap coefficient [P]_[P*]_coloc |	Overlap coefficient (overlap over union) for the current ROI for puncta masks detected from each pair of puncta channels [P] and [P*]. This is a measure of colocalization between puncta detected in different channels
