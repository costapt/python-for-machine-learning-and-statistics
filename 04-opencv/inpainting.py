import cv2

img = cv2.imread('inpainting1.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
mask = cv2.imread('inpainting2.jpg', 0)
mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

dst = cv2.inpaint(img, mask, 50, cv2.INPAINT_TELEA)

cv2.imwrite("inpainting3.jpg", dst)
