import os, importlib, sys
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
image_set = 'test'      #'train', 'test'
svm_experimentName = 'exp1'

#no need to change these parameters
boUseNonMaximaSurpression = True
visualizationDir = resultsDir + "visualizations"
cntkParsedOutputDir = cntkFilesDir + image_set + "_" + classifier + "_parsed/"



####################################
# Main
####################################
#load svm
print "classifier = " + classifier
makeDirectory(resultsDir)
makeDirectory(visualizationDir)
if classifier == "svm":
    print "Loading svm weights.."
    svmWeights, svmBias, svmFeatScale = loadSvm(trainedSvmDir, svm_experimentName)


#loop over all images and visualize
imdb = imdbs[image_set]
for imgIndex in range(0, imdb.num_images):
    imgPath = imdb.image_path_at(imgIndex)
    imgWidth, imgHeight = imWidthHeight(imgPath)

    #evaluate classifier for all rois
    if classifier == "svm":
        labels, scores = svmPredict(imgIndex, cntkParsedOutputDir, svmWeights, svmBias, svmFeatScale, cntk_nrRois, len(classes), vis_decisionThresholds[classifier])
    elif classifier == "nn":
        labels, scores = nnPredict(imgIndex, cntkParsedOutputDir, cntk_nrRois, len(classes), vis_decisionThresholds[classifier])
    else:
        ERROR

    #remove the zero-padded rois
    scores = scores[:len(imdb.roidb[imgIndex]['boxes'])]
    labels = labels[:len(imdb.roidb[imgIndex]['boxes'])]

    #perform non-maxima surpression. note that the detected classes in the image is not affected by this.
    nmsKeepIndices = []
    if boUseNonMaximaSurpression:
        nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, labels, scores, imdb.roidb[imgIndex]['boxes'])
        print "Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(len(nmsKeepIndices), len(labels), nmsThreshold)

    #visualize results
    imgDebug = visualizeResults(imgPath, labels, scores, imdb.roidb[imgIndex]['boxes'], cntk_padWidth, cntk_padHeight,
                                classes, nmsKeepIndices, boDrawNegativeRois=False, boDrawNmsRejectedRois=False)
    #imshow(imgDebug, waitDuration=0, maxDim = 800)
    imwrite(imgDebug, visualizationDir + "/" + classifier + "_" + str(imgIndex) + os.path.basename(imgPath))
print "DONE."
