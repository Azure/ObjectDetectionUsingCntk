import os, importlib, sys
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_set = 'test'      #'train', 'test'
svm_experimentName = 'exp1'

#no need to change these parameters
boIncludeGroundTruthRois = False   #remove GT (perfect) ROIs which were added to the 'train' imageSet
boUseNonMaximaSurpression = True
visualizationDir = resultsDir + "visualizations"
cntkParsedOutputDir = cntkFilesDir + image_set + "_" + classifier + "_parsed/"
if classifier == 'svm':
    prThresholds = np.linspace(0, 10, 21)
else:
    prThresholds = np.linspace(0, 1, 21)



####################################
# Main
####################################
#init
imdb     = imdbs[image_set]
gt_roidb = imdb.gt_roidb()
recalls    = collections.defaultdict(list)
precisions = collections.defaultdict(list)

#load svm
print("classifier = " + classifier)
makeDirectory(resultsDir)
makeDirectory(visualizationDir)
if classifier == "svm":
    print("Loading svm weights..")
    svmWeights, svmBias, svmFeatScale = loadSvm(trainedSvmDir, svm_experimentName)
else:
    svmWeights, svmBias, svmFeatScale = (None, None, None)


#loop over all images and visualize
for imgIndex in range(0, imdb.num_images):
    imgPath = imdb.image_path_at(imgIndex)
    imgWidth, imgHeight = imWidthHeight(imgPath)
    print("Processing image {} of {}: {}".format(imgIndex, imdb.num_images, imgPath))

    #load DNN output
    cntkOutputPath = os.path.join(cntkParsedOutputDir,  str(imgIndex) + ".dat.npz")
    dnnOutput = np.load(cntkOutputPath)['arr_0']
    assert(len(dnnOutput) == cntk_nrRois)

    #evaluate classifier for all rois and remove the zero-padded rois
    labels, scores = scoreRois(classifier, dnnOutput, svmWeights, svmBias, svmFeatScale, len(classes)) #, vis_decisionThresholds[classifier])
    scores = scores[:len(imdb.roidb[imgIndex]['boxes'])]
    labels = labels[:len(imdb.roidb[imgIndex]['boxes'])]

    #remove the ground truth ROIs which were added for training purposes
    if not boIncludeGroundTruthRois:
        inds = np.where(imdb.roidb[imgIndex]['gt_classes'] == 0)[0]
        labels = [labels[i] for i in inds]
        scores = [scores[i] for i in inds]
        imdb.roidb[imgIndex]['boxes'] = imdb.roidb[imgIndex]['boxes'][inds]

    #perform non-maxima surpression. note that the set of labels detected in the image is not affected by this.
    nmsKeepIndices = []
    if boUseNonMaximaSurpression:
        nmsKeepIndices = applyNonMaximaSuppression(nmsThreshold, labels, scores, imdb.roidb[imgIndex]['boxes'])
        print("Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(len(nmsKeepIndices), len(labels), nmsThreshold))

    #visualize results
    imgDebug = visualizeResults(imgPath, labels, scores, imdb.roidb[imgIndex]['boxes'], classes, nmsKeepIndices,
                                boDrawNegativeRois=False, boDrawNmsRejectedRois=False, decisionThreshold = vis_decisionThresholds[classifier])
    imshow(imgDebug, waitDuration=1, maxDim = 800)
    imwrite(imgDebug, visualizationDir + "/" + classifier + "_" + str(imgIndex) + os.path.basename(imgPath))


    #compute precision recall of the detection for different thresholds
    gtLabels = gt_roidb[imgIndex]['gt_classes']
    gtBboxes = [Bbox(*rect) for rect in gt_roidb[imgIndex]['boxes']]

    for thres in prThresholds:
        # get detections with scores higher than the threshold and which were kept by nms
        keepInds = set(np.where((np.array(labels) > 0) & (np.array(scores) > thres))[0])
        if boUseNonMaximaSurpression:
            keepInds = keepInds.intersection(nmsKeepIndices)
        detLabels = [labels[i] for i in keepInds]
        detBboxes = [Bbox(*imdb.roidb[imgIndex]['boxes'][i]) for i in keepInds]

        #compute precision recall of the detection
        precision, recall = detPrecisionRecall(detBboxes, detLabels, gtBboxes, gtLabels,
                                               evalVocOverlapThreshold, boPenalizeMultipleDetections=False)
        recalls[thres].append(recall)
        if precision != None:
            precisions[thres].append(precision)


#compute precision and recall at different thresholds
print("Precision/recall when rejecting detections below a given threshold:")
for thres in prThresholds:
    if precisions[thres] == []:
        break
    print("   At threshold {:.2f}: precision = {:2.2f}, recall = {:2.2f}".format(thres, np.mean(precisions[thres]), np.mean(recalls[thres])))

print("DONE.")