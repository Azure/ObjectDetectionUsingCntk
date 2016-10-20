import importlib
from fastRCNN.test import test_net
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_set = 'test'
svmExperimentName = "exp1"

#no need to change these
cntkParsedOutputDir = cntkFilesDir + image_set + "_" + classifier + "_parsed/"



####################################
# Main
####################################
print "classifier = " + classifier
imdb = imdbs[image_set]
net = DummyNet(4096, imdb.num_classes, cntkParsedOutputDir)

#load svm
svmFeatScale = None
if classifier == 'svm':
    svmWeights, svmBias, svmFeatScale = loadSvm(trainedSvmDir, svmExperimentName)
    net.params['cls_score'][0].data = svmWeights
    net.params['cls_score'][1].data = svmBias

#create empty directory for evaluation files
if type(imdb) == imdb_data:
    evalTempDir = None
else:
    #pascal_voc implementation requires temporary directory for evaluation
    evalTempDir = os.path.join(procDir, "eval_mAP_" + image_set)
    makeDirectory(evalTempDir)
    deleteAllFilesInDirectory(evalTempDir, None)

#compute mAPs
test_net(net, imdb, evalTempDir, svmFeatScale, classifier, nmsThreshold, boUsePythonImpl = True) #, boApplyNms = False) #, boThresholdDetections = False)

print "DONE."