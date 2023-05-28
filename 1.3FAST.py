# FAST  检测尺度不变换

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("./images/1.jpg")
fast = cv.FastFeatureDetector_create(threshold=30)
kp = fast.detect(img,None)
# img2 = cv.drawKeypoints(img,kp,None,color=(0,0,255))
img2 = cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img2[:,:,::-1])
plt.show()

# 关闭非极大值抑制
fast.setNonmaxSuppression(0)

kp = fast.detect(img,None)
# img3 = cv.drawKeypoints(img,kp,None,color=(0,0,255))
img3 = cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img3[:,:,::-1])
plt.show()

# cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # 图像显示
# plt.figure(figsize=(8,6),dpi=100)
# plt.imshow(img[:,:,::-1])