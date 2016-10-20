# -*- coding: utf-8 -*-
import cv2, os, sys, time, importlib
from Tkinter import *
from PIL import ImageTk
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
imagesToAnnotateDir = "C:/Users/pabuehle/Desktop/newImgs/"

#no need to change these
boxWidth = 10
boxHeight = 2
drawingMaxImgSize = 1000
objectNames = classes[1:]
objectNames = np.sort(objectNames).tolist()
objectNames += ["UNDECIDED", "EXCLUDE"]




####################################
# Helper functions
####################################
def buttonPressedCallback(s):
    global tkLastButtonPressed
    global tkBoButtonPressed
    tkLastButtonPressed = s
    tkBoButtonPressed = True



####################################
# Main
####################################
#create UI
tk = Tk()
w = Canvas(tk, width=len(objectNames) * boxWidth, height=len(objectNames) * boxHeight, bd = boxWidth, bg = 'white')
w.grid(row = len(objectNames), column = 0, columnspan = 2)
for objectIndex,objectName in enumerate(objectNames):
    b = Button(width=boxWidth, height=boxHeight, text=objectName, command=lambda s = objectName: buttonPressedCallback(s))
    b.grid(row = objectIndex, column = 0)


#loop over all images
imgFilenames = getFilesInDirectory(imagesToAnnotateDir, ".jpg")
for imgIndex, imgFilename in enumerate(imgFilenames):
    print imgIndex, imgFilename
    labelsPath = imagesToAnnotateDir + "/" + imgFilename[:-4] + ".bboxes.labels.tsv"
    if os.path.exists(labelsPath):
        continue

    #load image and bboxes
    imgPath = imagesToAnnotateDir + "/" + imgFilename
    print imgIndex, imgPath
    img = imread(imgPath)
    rectsPath = imgPath = imagesToAnnotateDir + "/" + imgFilename[:-4] + ".bboxes.tsv"
    rects = readTable(rectsPath)
    rects = [ToIntegers(rect) for rect in rects]

    #annotate each rectangle in turn
    labels = []
    for rectIndex,rect in enumerate(rects):
        imgCopy = img.copy()
        drawRectangles(imgCopy, [rect], thickness = 15)

        #draw image in tk window
        imgTk, _ = imresizeMaxDim(imgCopy, drawingMaxImgSize)
        imgTk = imconvertCv2Pil(imgTk)
        imgTk = ImageTk.PhotoImage(imgTk)
        label = Label(tk, image=imgTk)
        label.grid(row=0, column=1, rowspan=drawingMaxImgSize)
        tk.update_idletasks()
        tk.update()

        #busy-wait until button pressed
        tkBoButtonPressed = False
        tkLastButtonPressed = None
        while not tkBoButtonPressed:
            tk.update_idletasks()
            tk.update()

        #store result
        print "tkLastButtonPressed", tkLastButtonPressed
        labels.append(tkLastButtonPressed)

    writeFile(labelsPath, labels)
tk.destroy()
print "DONE."