import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# 加载图像
img0 = cv.imread('./images/3.jpg', cv.IMREAD_GRAYSCALE)
img1 = cv.imread('./images/4.jpg', cv.IMREAD_GRAYSCALE)

# 使用 ORB  融合了FAST关键点检测器和BRIEF关键点描述符
# orb = cv.ORB_create()
# kp0, des0 = orb.detectAndCompute(img0, None)
# kp1, des1 = orb.detectAndCompute(img1, None)

# 使用 sift
# sift = cv.SIFT_create()
# 关键点检测：kp关键点信息包括方向，尺度，位置信息，des是关键点的描述符
# kp0, des0 = sift.detectAndCompute(img0, None)
# kp1, des1 = sift.detectAndCompute(img1, None)
sift = cv.SIFT_create()
kp0 = sift.detect(img0, None)
kp1 = sift.detect(img1, None)

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp0, des0 = brief.compute(img0, kp0)
kp1, des1 = brief.compute(img1, kp1)

# 暴力匹配
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# bf = cv.BFMatcher(cv.NORM_L1)
matches = bf.match(des0, des1)

# 按照距离把匹配排序
matches = sorted(matches, key=lambda x:x.distance)

# 画出前25个匹配
img_matches = cv.drawMatches(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 显示匹配结果
plt.imshow(img_matches)
plt.show()
