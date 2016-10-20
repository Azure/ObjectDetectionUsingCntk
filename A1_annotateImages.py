import os, sys, importlib, shutil
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)



####################################
# Parameters
####################################
imagesToAnnotateDir = "C:/Users/pabuehle/Desktop/newImgs/"

#no need to change these params
drawingMaxImgSize = 1000.0
annotationsFile = resultsDir + "annotations.tsv"
minNrPixels = -1


####################################
# Functions
####################################
def event_cv2GetRectangles(event, x, y, flags, param):
    global cv2GetRectangle_global_bboxes
    global cv2GetRectangle_global_leftButtonDownPoint
    boLeftMouseDown = flags == cv2.EVENT_FLAG_LBUTTON

    #draw all previous bounding boxes
    imgCopy = image.copy()
    drawRectangles(imgCopy, cv2GetRectangle_global_bboxes)
    if len(cv2GetRectangle_global_bboxes)>0:
        drawRectangles(imgCopy, [cv2GetRectangle_global_bboxes[-1]], color = (255, 0, 0))

    #handle mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2GetRectangle_global_leftButtonDownPoint = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pt1 = cv2GetRectangle_global_leftButtonDownPoint
        pt2 = (x, y)
        minPt = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
        maxPt = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))
        imgWidth, imgHeight = imWidthHeight(image)
        minPt = ptClip(minPt, imgWidth, imgHeight)
        maxPt = ptClip(maxPt, imgWidth, imgHeight)
        cv2GetRectangle_global_bboxes.append(minPt + maxPt)
    elif boLeftMouseDown:
        cv2.rectangle(imgCopy, cv2GetRectangle_global_leftButtonDownPoint, (x, y), (255, 255, 0), 1)
    else:
        drawCrossbar(imgCopy, (x, y))
    cv2.imshow("image", imgCopy)

def procBoundingBoxes(rectsIn, imageUnscaled, scaleFactor):
    if len(rectsIn) <= 0:
        return rectsIn
    else:
        rects = copy.deepcopy(rectsIn)
        for index in range(len(rects)):
            for i in range(4):
                rects[index][i] = int(round(rects[index][i] / scaleFactor))
        imgWidth, imgHeight = imWidthHeight(imageUnscaled)
        bboxes = [Bbox(*rect) for rect in rects]
        for bbox in bboxes:
            bbox.crop(imgWidth, imgHeight)
            assert(bbox.isValid())
        return [bbox.rect() for bbox in bboxes]




####################################
# Main
####################################
makeDirectory(resultsDir)
imgFilenames = [f for f in os.listdir(imagesToAnnotateDir) if f.lower().endswith(".jpg")]

print "Using annotations file: " + annotationsFile
if annotationsFile and os.path.exists(annotationsFile):
    shutil.copyfile(annotationsFile, annotationsFile + ".backup.tsv")
    data = readTable(annotationsFile)
    annotationsLUT = getDictionary(getColumn(data,0), getColumn(data,1), False)
else:
    annotationsLUT = dict()


#loop over each image and get annotation
for imgFilenameIndex,imgFilename in enumerate(imgFilenames):
    print imgFilenameIndex,imgFilename
    imgPath = imagesToAnnotateDir + imgFilename
    print "Processing image {0} of {1}: {2}".format(imgFilenameIndex, len(imgFilenames), imgPath)
    bBoxPath = imgPath[:-4] + ".bboxes.tsv"

    #compute scale factor
    imgWidth, imgHeight = imWidthHeight(imgPath)
    scaleFactor = min(1, drawingMaxImgSize / max(imgWidth, imgHeight))
    if imgWidth * imgHeight < minNrPixels:
        print "Low resolution ({0},{1}) hence skipping image: {2}.".format(imgWidth, imgHeight, imgPath)
        continue

    #load existing ground truth if provided
    cv2GetRectangle_global_bboxes = []
    if os.path.exists(bBoxPath):
        print "Skipping image since ground truth already exists: %s." % imgPath
        continue

    #draw image
    imageUnscaled = imread(imgPath)
    image = imresize(imageUnscaled, scaleFactor)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", event_cv2GetRectangles)
    imgCopy = image.copy()
    drawRectangles(imgCopy, cv2GetRectangle_global_bboxes)
    cv2.imshow("image", imgCopy)


    #wait for user input
    while True:
        key = unichr(cv2.waitKey()) #& 0xFF

        #skip
        if key == "s":
            if os.path.exists(bBoxPath):
                print "Skipping image hence deleting existing bbox file: " + bBoxPath
                os.remove(bBoxPath)
            annotationsLUT[imgPath] = "skip"
            if annotationsFile:
                writeTable(annotationsFile, sortDictionary(annotationsLUT))
            break

        #undo
        if key == "u":
            if len(cv2GetRectangle_global_bboxes) >= 1:
                cv2GetRectangle_global_bboxes = cv2GetRectangle_global_bboxes[:-1]
                imgCopy = image.copy()
                drawRectangles(imgCopy, cv2GetRectangle_global_bboxes)
                cv2.imshow("image", imgCopy)

        #next image
        elif key == "n":
            bboxes = procBoundingBoxes(cv2GetRectangle_global_bboxes, imageUnscaled, scaleFactor)
            writeTable(bBoxPath, bboxes)
            annotationsLUT[imgPath] = bboxes
            if annotationsFile:
                writeTable(annotationsFile, sortDictionary(annotationsLUT))
            break

        #quit
        elif key == "q":
            sys.exit()

cv2.destroyAllWindows()
print "DONE."