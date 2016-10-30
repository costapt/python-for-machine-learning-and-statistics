from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

 # noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, 3, 5)

_, sure_fg = cv2.threshold(dist_transform, 0.7 * np.max(dist_transform),
                           255, 0)
sure_fg = np.uint8(sure_fg)

cv2.imshow("thresh", np.hstack((gray, thresh,
                                (255. * dist_transform /
                                 np.max(dist_transform)).astype(np.uint8),
                                 sure_fg)))

# Finding unknown region
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
plt.imshow(markers)
plt.show()

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
    
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(markers)
plt.show()
