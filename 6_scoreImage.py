# -*- coding: utf-8 -*-
import sys, os, importlib, random
import PARAMETERS
import json

locals().update(importlib.import_module("PARAMETERS").__dict__)

####################################
# Parameters
####################################
imgDir = r"C:/Users/pabuehle/Desktop/newImgs/"


#choose which classifier to use
classifier = 'svm'
svm_experimentName = 'exp1'

# no need to change these parameters
boAddSelectiveSearchROIs = True
boAddGridROIs = True
boFilterROIs = True
boUseNonMaximaSurpression = True


####################################
# Main
####################################
random.seed(0)

# load cntk model
print("Loading DNN..")
tstart = datetime.datetime.now()
model_path = os.path.join(modelDir, "frcn_" + classifier + ".model")
if not os.path.exists(model_path):
    raise Exception("Model {} not found.".format(model_path))
model = load_model(model_path)
print("Time loading DNN [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# load trained svm
if classifier == "svm":
    print("Loading svm weights..")
    tstart = datetime.datetime.now()
    svmWeights, svmBias, svmFeatScale = loadSvm(trainedSvmDir, svm_experimentName)
    print("Time loading svm [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))
else:
    svmWeights, svmBias, svmFeatScale = (None, None, None)


# loop over all images
imgPaths = [os.path.join(imgDir, s) for s in getFilesInDirectory(imgDir, ".jpg")]
for imgIndex, imgPath in enumerate(imgPaths):
    print("\nProcessing image {} of {}: {}".format(imgIndex, len(imgPaths), imgPath))

    # compute ROIs
    tstart = datetime.datetime.now()
    imgOrig = imread(imgPath)
    currRois = computeRois(imgOrig, boAddSelectiveSearchROIs, boAddGridROIs, boFilterROIs, ss_kvals, ss_minSize,
                       ss_max_merging_iterations, ss_nmsThreshold,
                       roi_minDimRel, roi_maxDimRel, roi_maxImgDim, roi_maxAspectRatio, roi_minNrPixelsRel,
                       roi_maxNrPixelsRel, grid_nrScales, grid_aspectRatios, grid_downscaleRatioPerIteration,
                       grid_stepSizeRel)
    currRois = currRois[:cntk_nrRois]  # only keep first cntk_nrRois rois
    print("Time roi computation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

    # prepare DNN inputs
    tstart = datetime.datetime.now()
    imgPadded = imresizeAndPad(imgOrig, cntk_padWidth, cntk_padHeight)
    _, _, roisCntk = getCntkInputs(imgOrig, currRois, None, train_posOverlapThres, nrClasses, cntk_nrRois, cntk_padWidth, cntk_padHeight)
    arguments = {
        model.arguments[0]: [np.ascontiguousarray(np.array(imgPadded, dtype=np.float32).transpose(2, 0, 1))], # convert to CNTK's HWC format
        model.arguments[1]: [np.array(roisCntk, dtype=np.float32)]
    }
    print("Time cnkt input generation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

    # run DNN model
    print("Running model..")
    tstart = datetime.datetime.now()
    dnnOutputs = model.eval(arguments)[0]
    dnnOutputs = dnnOutputs[:len(currRois)]  # remove the zero-padded rois
    print("Time running model [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

    # score all ROIs
    tstart = datetime.datetime.now()
    labels, scores = scoreRois(classifier, dnnOutputs, svmWeights, svmBias, svmFeatScale, len(classes),
                               decisionThreshold = vis_decisionThresholds[classifier])
    print("Time making prediction [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

    # perform non-maxima surpression
    tstart = datetime.datetime.now()
    nmsKeepIndices = []
    if boUseNonMaximaSurpression:
        nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, labels, scores, currRois)
        print("Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(
            len(nmsKeepIndices), len(labels), nmsThreshold))
    print("Time non-maxima surpression [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

    # visualize results
    imgDebug = visualizeResults(imgPath, labels, scores, currRois, classes, nmsKeepIndices,
                                boDrawNegativeRois=False, boDrawNmsRejectedRois=False)
    imshow(imgDebug, waitDuration=0, maxDim=800)

    # create json-encoded string of all detections
    outDict = [{"label": str(l), "score": str(s), "nms": str(False), "left": str(r[0]), "top": str(r[1]), "right": str(r[2]), "bottom": str(r[3])} for l,s, r in zip(labels, scores, currRois)]
    for i in nmsKeepIndices:
        outDict[i]["nms"] = str(True)
    outJsonString = json.dumps(outDict)
    print("Json-encoded detections: " + outJsonString[:120] + "...")

print("DONE.")



#--- optional code ---#

# write all detections to file, and show how to read in again to visualize
# writeDetectionsFile("detections.tsv", outDict, classes)
# labels2, scores2, currRois2, nmsKeepIndices2 = parseDetectionsFile("detections.tsv", lutClass2Id)
# imgDebug2 = visualizeResults(imgPath, labels2, scores2, currRois2, classes, nmsKeepIndices2,  # identical to imgDebug
#                              boDrawNegativeRois=False, boDrawNmsRejectedRois=False)
# imshow(imgDebug2, waitDuration=0, maxDim=800)

# extract crop of the highest scored ROI
# maxScore = -float("inf")
# maxScoreRoi = []
# for index, (label,score) in enumerate(zip(labels,scores)):
#    if score > maxScore and label > 0: #and index in nmsKeepIndices:
#        maxScore = score
#        maxScoreRoi = currRois[index]
# if maxScoreRoi == []:
#    print("WARNING: not a single object detected")
# else:
#    imgCrop = imgOrig[maxScoreRoi[1]:maxScoreRoi[3], maxScoreRoi[0]:maxScoreRoi[2], :]
#    imwrite(imgCrop, outCropDir + os.path.basename(imgPath))
#    imshow(imgCrop)

