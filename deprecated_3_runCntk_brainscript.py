import os, sys, importlib
import shutil, time
import subprocess
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
cntkBinariesDir = "C:/local/CNTK-2-0-rc1/cntk/cntk/"

# no need to change this
cntkCmdStrPattern = "{0}/cntk.exe configFile={1}config.cntk currentDirectory={1}"



####################################
# Main
####################################
print("classifier = " + classifier)
if not os.path.exists(cntkBinariesDir + "/cntk.exe"):
    raise Exception("Cannot find cntk.exe in directory: " + cntkBinariesDir)
deleteAllFilesInDirectory(cntkFilesDir + "/tmp", None)
shutil.copy(os.path.join(cntkResourcesDir, "config.cntk"), cntkFilesDir)

#generate cntk command string
cmdStr = cntkCmdStrPattern.format(cntkBinariesDir, cntkFilesDir, classifier)
cmdStr += " ImageH={} ImageW={}".format(cntk_padHeight, cntk_padWidth)
cmdStr += " NumLabels={0} NumTrainROIs={1} NumTestROIs={1}".format(len(classes), cntk_nrRois)
cmdStr += " TrainROIDim={} TrainROILabelDim={}".format(4*cntk_nrRois, cntk_nrRois * cntk_featureDimensions[classifier])
cmdStr += " TestROIDim={} TestROILabelDim={}".format(  4*cntk_nrRois, cntk_nrRois * cntk_featureDimensions[classifier])
if classifier == 'svm':
    cmdStr += " [Train=[SGD=[maxEpochs=0]]]" #no need to train the network if just using it as featurizer
    cmdStr += " [WriteTest=[outputNodeNames=(z.fcOut.h2.y)]]"
    cmdStr += " [WriteTrain=[outputNodeNames=(z.fcOut.h2.y)]]"

#run cntk
tstart = datetime.datetime.now()
os.environ['ACML_FMA'] = str(0)
print(cmdStr)
pid = subprocess.Popen(cmdStr, cwd = cntkFilesDir) #, creationflags=subprocess.CREATE_NEW_CONSOLE)
pid.wait()
print ("Time running cntk [s]: " + str((datetime.datetime.now() - tstart).total_seconds()))

#delete model files written during cntk training
filenames = getFilesInDirectory(cntkFilesDir + "/tmp/", postfix = None)
for filename in filenames:
    if filename.startswith('Fast-RCNN.'):
        os.remove(cntkFilesDir + "/tmp/" + filename)
assert pid.returncode == 0, "ERROR: cntk ended with exit code {}".format(pid.returncode)

#parse cntk output
print("classifier = " + classifier)
image_sets = ["test", "train"]
for image_set in image_sets:
    print("Parsing CNTK output for image set: " + image_set)
    cntkImgsListPath = cntkFilesDir + image_set + ".txt"
    outParsedDir = cntkFilesDir + image_set + "_" + classifier + "_parsed/"
    if classifier == 'svm':
        cntkOutputPath = cntkFilesDir + image_set + ".z.fcOut.h2.y"
    elif classifier == 'nn':
        cntkOutputPath = cntkFilesDir + image_set + ".z"
    else:
        error

    #write cntk output for each image to separate file
    makeDirectory(outParsedDir)
    parseCntkOutput(cntkImgsListPath, cntkOutputPath, outParsedDir, cntk_nrRois, cntk_featureDimensions[classifier],
                    saveCompressed = True, skipCheck = False) #, skip5Mod = 0)

    #delete cntk output file which can be very large and are no longer needed
    deleteFile(cntkOutputPath)
print("DONE.")