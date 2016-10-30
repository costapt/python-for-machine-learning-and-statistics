import numpy as np
import cv2

camera = cv2.VideoCapture('video.avi')
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    grabbed, frame = camera.read()

    if not grabbed:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    fgmask = mog.apply(frame, learningRate=0.001)

    fgmask = cv2.erode(fgmask, (7, 7), iterations=3)
    fgmask = cv2.dilate(fgmask, (7, 7), iterations=3)

    cv2.imshow("fgbg", np.hstack((gray, fgmask)))

    if cv2.waitKey(30) != -1:
        break
