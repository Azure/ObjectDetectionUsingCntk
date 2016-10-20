# -*- coding: utf-8 -*-
import sys, os, importlib, random
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
boSaveDebugImg = True
subdirs = ['positive', 'testImages', 'negative']

#no need to change these parameters
boAddSelectiveSearchROIs = True
boAddRoisOnGrid = True
boFilterRois = True


####################################
# Main
####################################
#init
makeDirectory(roiDir)
roi_minDim = roi_minDimRel * roi_maxImgDim
roi_maxDim = roi_maxDimRel * roi_maxImgDim
roi_minNrPixels = roi_minNrPixelsRel * roi_maxImgDim*roi_maxImgDim
roi_maxNrPixels = roi_maxNrPixelsRel * roi_maxImgDim*roi_maxImgDim

for subdir in subdirs:
    makeDirectory(roiDir + subdir)
    imgFilenames = getFilesInDirectory(imgDir + subdir, ".jpg")

    #loop over all images
    for imgIndex,imgFilename in enumerate(imgFilenames):
        roiPath = "{}/{}/{}.roi.txt".format(roiDir, subdir, imgFilename[:-4])
        #if os.path.exists(roiPath):
        #    print "Skipping image since roi file already exists: " + imgFilename, imgIndex
        #    continue

        # load image
        print imgIndex, len(imgFilenames), subdir, imgFilename
        tstart = datetime.datetime.now()
        imgPath = imgDir + subdir + "/" + imgFilename
        imgOrig = imread(imgPath)
        if imWidth(imgPath) > imHeight(imgPath):
            print imWidth(imgPath) , imHeight(imgPath)

        # get rois
        if boAddSelectiveSearchROIs:
            print "Calling selective search.."
            rects, img, scale = getSelectiveSearchRois(imgOrig, ss_scale, ss_sigma, ss_minSize, roi_maxImgDim) #interpolation=cv2.INTER_AREA
            print "   Number of rois detected using selective search: " + str(len(rects))
        else:
            rects = []
            img, scale = imresizeMaxDim(imgOrig, roi_maxImgDim, boUpscale=True, interpolation=cv2.INTER_AREA)
        imgWidth, imgHeight = imWidthHeight(img)

        # add grid rois
        if boAddRoisOnGrid:
            rectsGrid = getGridRois(imgWidth, imgHeight, grid_nrScales, grid_aspectRatios, grid_downscaleRatioPerIteration)
            print "   Number of rois on grid added: " + str(len(rectsGrid))
            rects += rectsGrid

        #run filter
        if not boFilterRois:
            rois = rects
        else:
            print "   Number of rectangles before filtering  = " + str(len(rects))
            rois = filterRois(rects, imgWidth, imgHeight, roi_minNrPixels, roi_maxNrPixels, roi_minDim, roi_maxDim, roi_maxAspectRatio)
            if len(rois) == 0: #make sure at least one roi returned per image
                rois = [[5, 5, imgWidth-5, imgHeight-5]]
            print "   Number of rectangles after filtering  = " + str(len(rois))

        #scale up to original size and save to disk
        #note: each rectangle is in original image format with [x,y,x2,y2]
        rois = np.int32(np.array(rois) / scale)
        assert (np.min(rois) >= 0)
        assert (np.max(rois[:, [0,2]]) < imWidth(imgOrig))
        assert (np.max(rois[:, [1,3]]) < imHeight(imgOrig))
        np.savetxt(roiPath, rois, fmt='%d')
        print "   Time [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000)

        #visualize detection
        if True or boSaveDebugImg:
            debugScale = 800.0 / max(imWidthHeight(imgOrig))
            img = imresize(imgOrig, debugScale)
            drawRectangles(img, rois*debugScale, color=(0, 255, 0), thickness=1)
            imshow(img, waitDuration = 1)
            if boSaveDebugImg:
                roiImgPath = os.path.join(roiDir, subdir, imgFilename[:-4] + ".roi.jpg")
                imwrite(img, roiImgPath)

print "DONE."