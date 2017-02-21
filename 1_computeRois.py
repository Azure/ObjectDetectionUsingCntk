# -*- coding: utf-8 -*-
import sys, os, importlib, random
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
boShowImg = True
subdirs = ['positive', 'testImages', 'negative']

#no need to change these parameters
boAddSelectiveSearchROIs = True
boAddGridROIs = True
boFilterROIs = True
if datasetName.lower() == "pascalvoc":
    print("No need to run ROI computation since Pascal VOC comes with pre-computed ROIs.")
    exit()


####################################
# Main
####################################
#init
for subdir in subdirs:
    makeDirectory(roiDir)
    makeDirectory(roiDir + subdir)
    imgFilenames = getFilesInDirectory(imgDir + subdir, ".jpg")

    #loop over all images
    times = []
    for imgIndex, imgFilename in enumerate(imgFilenames):
        #if os.path.exists(roiPath):
        #    print "Skipping image since roi file already exists: " + imgFilename, imgIndex
        #    continue

        # load image
        print("Processing image {} of {}: subdir={}, filename={}".format(imgIndex, len(imgFilenames), subdir, imgFilename))
        imgPath = join(imgDir, subdir, imgFilename)
        imgOrig = imread(imgPath)

        # compute ROIs
        tstart = datetime.datetime.now()
        rois = computeRois(imgOrig, boAddSelectiveSearchROIs, boAddGridROIs, boFilterROIs, ss_kvals, ss_minSize, ss_max_merging_iterations, ss_nmsThreshold,
                           roi_minDimRel, roi_maxDimRel, roi_maxImgDim, roi_maxAspectRatio, roi_minNrPixelsRel, roi_maxNrPixelsRel,
                           grid_nrScales, grid_aspectRatios, grid_downscaleRatioPerIteration)
        times.append((datetime.datetime.now() - tstart).total_seconds() * 1000)
        print("   Time roi computation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))
        roiPath = "{}/{}/{}.roi.txt".format(roiDir, subdir, imgFilename[:-4])
        np.savetxt(roiPath, rois, fmt='%d')

        #visualize ROIs
        if boShowImg:
            debugScale = 800.0 / max(imWidthHeight(imgOrig))
            img = imresize(imgOrig, debugScale)
            drawRectangles(img, rois*debugScale, color=(0, 255, 0), thickness=1)
            imshow(img, waitDuration = 1)
            roiImgPath = os.path.join(roiDir, subdir, imgFilename[:-4] + ".roi.jpg")
            imwrite(img, roiImgPath)

    print("Time per image [ms]: median={:.1f}, std={:.1f}, 90%-percentile={:.1f}".format(np.median(times), np.std(times), np.percentile(times, 90)))
print("DONE.")