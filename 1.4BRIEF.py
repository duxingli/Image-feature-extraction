import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("./images/1.jpg")

# 初始化 FAST
star = cv.xfeatures2d.StarDetector_create()
# 初始化 BRIEF
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# 找到 keypoints
kp = star.detect(img, None)
# print("11",kp)
# 计算 BRIEF的描述符
kp, des = brief.compute(img, kp)
# print("22",kp)
img2 = cv.drawKeypoints(img,kp,None,flags=0)
# img2 = cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# print(brief.descriptorSize())
# print(des.shape)
# print(des)
plt.imshow(img2[:,:,::-1])
plt.show()

