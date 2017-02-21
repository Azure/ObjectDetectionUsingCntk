import os, sys, importlib
import shutil, time
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
image_sets = ["train", "test"]


####################################
# Main
####################################
#clear imdb cache and other files
if os.path.exists(cntkFilesDir):
    assert(cntkFilesDir.endswith("cntkFiles/"))
    userInput = input('--> INPUT: Press "y" to delete directory ' + cntkFilesDir + ": ")
    if userInput.lower() not in ['y', 'yes']:
        print("User input is %s: exiting now." % userInput)
        exit(-1)
    shutil.rmtree(cntkFilesDir)
    time.sleep(0.2) #avoid file access errors


#create cntk representation for each image
makeDirectory(cntkFilesDir)
for image_set in image_sets:
    imdb = imdbs[image_set]
    counterGt = np.zeros(len(classes), np.int32)
    print("Number of images in set '{}' = {}".format(image_set, imdb.num_images))

    #open files for writing
    cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, nrRoisPath = cntkInputPaths(cntkFilesDir, image_set)
    with open(cntkImgsPath, 'w')      as cntkImgsFile, \
         open(cntkRoiCoordsPath, 'w') as cntkRoiCoordsFile, \
         open(cntkRoiLabelsPath, 'w') as cntkRoiLabelsFile, \
         open(nrRoisPath, 'w')        as nrRoisFile:

            # for each image, transform rois etc to cntk format
            for imgIndex in range(0, imdb.num_images):
                if imgIndex % 200 == 0:
                    print("Processing image set '{}', image {} of {}".format(image_set, imgIndex, imdb.num_images))
                imgPath = imdb.image_path_at(imgIndex)
                currRois = imdb.roidb[imgIndex]['boxes']
                currGtOverlaps = imdb.roidb[imgIndex]['gt_overlaps']
                for i in imdb.roidb[imgIndex]['gt_classes']:
                    counterGt[i] += 1

                #get DNN inputs for image
                #Note: this also marks other ROIs as 'positives', if overlap with GT is above a threshold
                labelsStr, roisStr, _ = getCntkInputs(imgPath, currRois, currGtOverlaps, train_posOverlapThres, nrClasses, cntk_nrRois, cntk_padWidth, cntk_padHeight)

                #update cntk data
                nrRoisFile.write("{}\n".format(len(currRois)))
                cntkImgsFile.write("{}\t{}\t0\n".format(imgIndex, imgPath))
                cntkRoiCoordsFile.write("{} |rois{}\n".format(imgIndex, roisStr))
                cntkRoiLabelsFile.write("{} |roiLabels{}\n".format(imgIndex, labelsStr))

    #print debug info
    if image_set == 'train':
        for i in range(len(classes)):
            print("   {:3}: Found {} objects of class {}.".format(i, counterGt[i], classes[i]))

print("DONE.")
