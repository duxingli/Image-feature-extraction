# ORB

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("./images/1.jpg")
orb = cv.ORB_create(nfeatures=5000)
# kp,des = orb.detectAndCompute(img,None)

kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
img2 = cv.drawKeypoints(img,kp,None,flags=0)
# img2 = cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img2[:,:,::-1])
plt.show()
# cv.imshow("img2",img2)
# cv.waitKey(0)