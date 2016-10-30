from matplotlib import pyplot as plt
import cv2
import os

tr_img = os.sys.argv[1]
q_img = os.sys.argv[2]

try:
    thrs = float(os.sys.argv[3])
except:
    thrs = 0.7

img1 = cv2.imread(tr_img, 0) # trainImage
img2 = cv2.imread(q_img, 0)  # queryImage

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

dst = cv2.drawKeypoints(img1, kp1, None,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure()
plt.imshow(dst)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = [[m] for m, n in matches if m.distance < thrs * n.distance]
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

plt.figure()
plt.imshow(img3))
plt.show()
