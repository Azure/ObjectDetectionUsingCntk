# -*- coding: utf-8 -*-
import sys, os, importlib
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
subdirs = ['positive']


####################################
# Main
####################################
overlaps = []
roiCounts = []
for subdir in subdirs:
    imgFilenames = getFilesInDirectory(imgDir + subdir, ".jpg")

    #loop over all iamges
    for imgIndex,imgFilename in enumerate(imgFilenames):
        if imgIndex % 50 == 0:
            print("Processing subdir '{}', image {} of {}".format(subdir, imgIndex, len(imgFilenames)))
        # load ground truth
        imgPath = imgDir + subdir + "/" + imgFilename
        imgWidth, imgHeight = imWidthHeight(imgPath)
        gtRois, gtLabels = readGtAnnotation(imgPath)
        gtRois = [Bbox(*roi) for roi in gtRois]

        # load rois and compute scale
        rois = readRois(roiDir, subdir, imgFilename)
        rois = rois[:cntk_nrRois] # only use the first N rois (similar to rest of code)
        rois = [Bbox(*roi) for roi in rois]
        roiCounts.append(len(rois))

        # for each ground truth, compute if it is covered by an roi
        for gtIndex, (gtLabel, gtRoi) in enumerate(zip(gtLabels,gtRois)):
            maxOverlap = -1
            assert (gtRoi.max() <= max(imgWidth, imgHeight) and gtRoi.max() >= 0)
            if gtLabel in classes[1:]:
                for roi in rois:
                    assert (roi.max() <= max(imgWidth, imgHeight) and roi.max() >= 0)
                    overlap = bboxComputeOverlapVoc(gtRoi, roi)
                    maxOverlap = max(maxOverlap, overlap)
            overlaps.append(maxOverlap)
print("Average number of rois per image " + str(int(1.0 * sum(roiCounts) / len(imgFilenames))))

#compute recall at different overlaps
recalls = []
overlaps = np.array(overlaps, np.float32)
for overlapThreshold in np.linspace(0,1,21):
    recall = 1.0 * sum(overlaps >= overlapThreshold) / len(overlaps)
    recalls.append(recall)
    print("At threshold {:.2f}: recall = {:2.2f}".format(overlapThreshold, recall))
print("Mean recall = {:2.2}".format(np.mean(recalls)))