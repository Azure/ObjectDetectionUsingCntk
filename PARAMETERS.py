from cntk_helpers import *
from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc # as nmsPython
print datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')



############################
# Adjust these parameters
# to make scripts run
############################
rootDir = os.path.dirname(os.path.realpath(sys.argv[0]))
cntkBinariesDir = "C:/local/cntk/cntk/"


############################
# default parameters
############################
datasetName = "grocery"  #"grocery", "pascalVoc"

#directories
imgDir = rootDir + "/data/" + datasetName + "/"
procDir = rootDir + "/proc/" + datasetName + "/"
resultsDir = rootDir + "/results/" + datasetName + "/"
roiDir = procDir + "rois/"
cntkFilesDir = procDir + "cntkFiles/"
trainedSvmDir = procDir + "trainedSvm/"
cntkResourcesDir = rootDir + "/resources/cntk/"

# ROI generation
roi_minDimRel = 0.01      # minimum relative width/height of a ROI
roi_maxDimRel = 1.0       # maximum relative width/height of a ROI
roi_minNrPixelsRel = 0    # minimum relative area covered by a ROI
roi_maxNrPixelsRel = 1.0  # maximum relative area covered by a ROI
roi_maxAspectRatio = 4.0  # maximum aspect Ratio of a ROI, both vertically and horizontally
roi_maxImgDim = 200       # image size used for ROI generation
ss_scale = 100            # selective search ROIs: parameter controlling cluster size for segmentation
ss_sigma = 1.2            # selective search ROIs: width of gaussian kernel for segmentation
ss_minSize = 20           # selective search ROIs: minimum component size for segmentation
grid_aspectRatios = [1.0, 2.0, 0.5]    # uniform grid ROIs: allowed aspect ratio of ROIs
grid_nrScales = 7                      # uniform grid ROIs: number of iterations from largest possible ROI to smaller ROIs
grid_downscaleRatioPerIteration = 1.5  # uniform grid ROIs: relative ROI width/height reduction per iteration, starting from largest possible ROI

# cntk model
cntk_nrRois = 2000        # DNN input number of ROIs per image. Zero-padded/truncated if necessary
cntk_padWidth = 1000      # DNN input image width [pixels]
cntk_padHeight = 1000     # DNN input image height [pixels]
cntk_featureDimensions = {'svm': 4096} # DNN output, dimension of each ROI

# nn and svm training
classifier = 'svm'        # Options: 'svm', 'nn'. Train either a Support Vector Machine, or directly the Neural Network
train_posOverlapThres=0.5 # DNN and SVM threshold for marking ROIs with significant overlap with a GT object as positive

# svm training
svm_C = 0.001             # regularization parameter of the soft-margin error term
svm_B = 10.0              # intercept scaling
svm_nrEpochs = 5          # number of training iterations
svm_retrainLimit = 2000   # number of new items to trigger SVM training
svm_evictThreshold = -1.1 # remove easy negatives with decision value below this threshold
svm_posWeight = "auto"    # automatically balance training set to correct for the majority of ROIs being negative
svm_targetNorm = 20.0     # magic value from traditional R-CNN (helps with convergence)
svm_penality = 'l2'       # penalty norm
svm_loss = 'l1'           # loss norm
svm_rngSeed = 3           # seed for randomization

# nn training
# All DNN training parameters are defined in the model definition brainscript file

# postprocessing
nmsThreshold = 0.3                      # Non-Maxima suppression threshold (in range [0,1])
                                        # The lower the more ROIs will be combined. Used during evaluation and visualization (scripts 5_)
vis_decisionThresholds = {'svm' : 0.5,  # Reject detections with low confidence, used only in 5_visualizeResults
                          'nn' : None}


############################
# project-specific parameters
############################
if datasetName.startswith("grocery"):
    classes = ('__background__',  # always have '__background__' be at index 0
               "avocado", "orange", "butter", "champagne", "eggBox", "gerkin", "joghurt", "ketchup",
               "orangeJuice", "onion", "pepper", "tomato", "water", "milk", "tabasco", "mustard")

    # roi generation
    roi_minDimRel = 0.04
    roi_maxDimRel = 0.4
    roi_minNrPixelsRel = 2    * roi_minDimRel * roi_minDimRel
    roi_maxNrPixelsRel = 0.33 * roi_maxDimRel * roi_maxDimRel

    # postprocessing
    nmsThreshold = 0.01

    # database
    imdbs = dict()      # database provider of images and image annotations
    for image_set in ["train", "test"]:
        imdbs[image_set] = imdb_data(image_set, classes, cntk_nrRois, imgDir, roiDir, cntkFilesDir, boAddGroundTruthRois = (image_set!='test'))


elif datasetName.startswith("pascalVoc"):
    classes = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    lutImageSet = {"train": "trainval", "test": "test"}

    # model training / scoring
    classifier = 'nn'

    #more than 99% of the test images have less than 4000 rois, but 50% more than 2000
    cntk_nrRois = 4000

    # database
    imdbs = dict()
    for image_set, year in zip(["train", "test"], ["2007", "2007"]):
        imdbs[image_set] = fastRCNN.pascal_voc(lutImageSet[image_set], year, classes, cntk_nrRois, cacheDir = cntkFilesDir)
        print "Number of {} images: {}".format(image_set, imdbs[image_set].num_images)

else:
     ERROR


############################
# computed parameters
############################
nrClasses = len(classes)
cntk_featureDimensions['nn'] = nrClasses

print "PARAMETERS: datasetName = " + datasetName
assert cntk_padWidth == cntk_padHeight, "ERROR: different width and height for padding not supported."
assert classifier.lower() in ['svm','nn'], "ERROR: only 'nn' or 'svm' classifier supported."
assert not (datasetName == 'pascalVoc' and classifier == 'svm'), "ERROR: 'svm' classifier for pascal VOC not supported."
assert(train_posOverlapThres >= 0 and train_posOverlapThres <= 1)