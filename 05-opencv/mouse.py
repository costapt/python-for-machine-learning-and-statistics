import numpy as np
import cv2
import os


def draw(event, x, y, flags, param):
    global img, mask, drawing, erasing, radius, prev_inpainting

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        dst = np.copy(img)
        if drawing or erasing:
            color = 255 * drawing
            cv2.circle(mask, (x, y), radius, color, -1)
            dst[mask != 0] = [0, 255, 0]
        else:
            dst = np.copy(prev_inpainting)
            cv2.circle(dst, (x, y), radius, [0, 0, 255], -1)
        cv2.imshow("img", dst)
    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        drawing = False
        erasing = False
        dst = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        prev_inpainting = np.copy(dst)
        cv2.imshow("img", dst)


def change_radius(x):
    global radius
    radius = x


img = cv2.imread('inpainting1.jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

mask = np.zeros(img.shape[: 2], dtype=np.uint8)
prev_inpainting = np.copy(img)
drawing = False
erasing = False
radius = 5

cv2.namedWindow("img")
cv2.createTrackbar('Radius','img', 5, 20, change_radius)
cv2.setMouseCallback("img", draw)

cv2.imshow("img", img)

while True:
    if cv2.waitKey(33) != -1:
        break
