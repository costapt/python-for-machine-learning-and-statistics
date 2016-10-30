import numpy as np
import cv2


def nothing(x):
    global img
    _, thrs = cv2.threshold(img, x, 255, 0)
    cv2.imshow('frame', thrs)


#creates three trackbars for color change
cv2.namedWindow('frame')
cv2.createTrackbar('Threshold','frame', 128, 255, nothing)

img = cv2.imread('receipt.png', 0)

nothing(128)
cv2.waitKey(0)
