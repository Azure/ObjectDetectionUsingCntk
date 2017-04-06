from PARAMETERS import *
from helpers_cntk import *


####################################
# MAIN
####################################
makeDirectory(modelDir)
print ("classifier = " + classifier)
print ("cntk_lr_per_image = " + str(cntk_lr_per_image))

# optionally retrain DNN
# if the classifier is svm, then simply return the 4096-floats penultimate layer as model
# otherwise add new output layer, retrain the DNN, and return this new model.
if classifier == 'svm':
    boSkipTraining = True
else:
    boSkipTraining = False
model = init_train_fast_rcnn(cntk_padHeight, cntk_padWidth, nrClasses, cntk_nrRois, cntk_mb_size, cntk_max_epochs,
                             cntk_lr_per_image, cntk_l2_reg_weight, cntk_momentum_time_constant, cntkFilesDir, boSkipTraining)

# write model to disk
model_path = os.path.join(modelDir, "frcn_" + classifier + ".model")
print("Writing model to %s" % model_path)
model.save_model(model_path)

# compute output of every image and write to disk
image_sets = ["test", "train"]
for image_set in image_sets:
    outParsedDir = cntkFilesDir + image_set + "_" + classifier + "_parsed/"
    makeDirectory(outParsedDir)
    run_fast_rcnn(model, image_set, cntk_padHeight, cntk_padWidth, nrClasses, cntk_nrRois, cntkFilesDir, outParsedDir)

print("DONE.")