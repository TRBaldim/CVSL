
from libs import *
from skimage.filters import threshold_adaptive
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import imutils


onlyfiles = [f for f in listdir('/home/thiago/CVSL/cars') if isfile(join('/home/thiago/CVSL/cars', f))]

for i in onlyfiles:

    image = cv2.imread('/home/thiago/CVSL/cars/' + i)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cv2.imshow("image", gray)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break


    cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 251, offset=10)
    warped = warped.astype('uint8') * 255

    cv2.imshow('Original', imutils.resize(orig, height=650))
    cv2.imshow('Scanned', imutils.resize(warped, height=300))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
