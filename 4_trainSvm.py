import importlib
from fastRCNN.train_svms import SVMTrainer
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


#################################################
# Parameters
#################################################
experimentName = "exp1"

#no need to change these params
cntkParsedOutputDir = cntkFilesDir + "train_svm_parsed/"



#################################################
# Main
#################################################
if classifier == "nn":
    print("No need to train SVM since using 'nn' classifier.")
    exit()
print ("svm_targetNorm = " + str(svm_targetNorm))
print ("svm_retrainLimit = " + str(svm_retrainLimit))
print ("svm_posWeight = " + str(svm_posWeight))
print ("svm_C = " + str(svm_C))
print ("svm_B = " + str(svm_B))
print ("svm_penality = " + str(svm_penality))
print ("svm_loss = " + str(svm_loss))
print ("svm_evictThreshold = " + str(svm_evictThreshold))
print ("svm_nrEpochs = " + str(svm_nrEpochs))

#init
makeDirectory(trainedSvmDir)
np.random.seed(svm_rngSeed)
imdb = imdbs["train"]
net = DummyNet(4096, imdb.num_classes, cntkParsedOutputDir)
svmWeightsPath, svmBiasPath, svmFeatScalePath = svmModelPaths(trainedSvmDir, experimentName)

# add ROIs which significantly overlap with a ground truth object as positives
if train_posOverlapThres > 0:
    print ("Adding ROIs with gt overlap >= %2.2f as positives ..." % (train_posOverlapThres))
    existingPosCounter, addedPosCounter = updateRoisGtClassIfHighGtOverlap(imdb, train_posOverlapThres)
    print ("Number of positives originally: {} (in {} images)".format(existingPosCounter, imdb.num_images))
    print ("Number of additional positives: {}.".format(addedPosCounter))

# start training
svm = SVMTrainer(net, imdb, im_detect, svmWeightsPath, svmBiasPath, svmFeatScalePath,
                 svm_C, svm_B, svm_nrEpochs, svm_retrainLimit, svm_evictThreshold, svm_posWeight,
                 svm_targetNorm, svm_penality, svm_loss, svm_rngSeed)
svm.train()
print ("DONE.")
