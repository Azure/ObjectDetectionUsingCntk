import os, sys, importlib
import shutil, time
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_set = "train"


####################################
# Main
####################################
# read ground truth and ROIs
if not os.path.exists(cntkFilesDir + image_set + ".cache_gt_roidb.pkl"):
    raise Exception("Run 2_cntkGenerateInputs.py before executing this script.")
imdb = imdbs[image_set]
gtRois = imdb.gt_roidb()
print("Number of images in set '{}' = {}".format(image_set, imdb.num_images))

# extract width, height, etc for all ground truth annotations in all images
roiInfos = []
for imgIndex in range(0, imdb.num_images):
    imgPath = imdb.image_path_at(imgIndex)
    imgWidth, imgHeight = imWidthHeight(imgPath)

    if gtRois[imgIndex] != None:
        for gtRoi in gtRois[imgIndex]['boxes']:
            roiWidth  = gtRoi[2] - gtRoi[0] +1
            roiHeight = gtRoi[3] - gtRoi[1] +1
            roiRelWidth  = float(roiWidth)  / imgWidth
            roiRelHeight = float(roiHeight) / imgHeight
            roiInfos.append((roiRelWidth, roiRelHeight, roiRelWidth * roiRelHeight, roiRelWidth / roiRelHeight))

# analyse typical width, height, etc of the ground truth annotations
print("\nStatistics for ground truth annotations:")
for percentile in np.linspace(0, 100, 21):
    print("   Percentile {:3.0f}: width = {:<.2f}, height = {:<.2f}, area = {:<.3f}, aspectRatio = {:<.2f}".format(
            percentile,
            np.percentile(getColumn(roiInfos, 0), percentile),
            np.percentile(getColumn(roiInfos, 1), percentile),
            np.percentile(getColumn(roiInfos, 2), percentile),
            np.percentile(getColumn(roiInfos, 3), percentile)))
print("DONE.")
