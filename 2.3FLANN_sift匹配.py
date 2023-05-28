import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img0 = cv.imread('./images/3.jpg',cv.IMREAD_GRAYSCALE)
img1 = cv.imread('./images/4.jpg',cv.IMREAD_GRAYSCALE)

#  SIFT 特征检测和描述.
# sift = cv.SIFT_create()
# kp0, des0 = sift.detectAndCompute(img0, None)
# kp1, des1 = sift.detectAndCompute(img1, None)

# orb = cv.ORB_create()
# kp0, des0 = orb.detectAndCompute(img0, None)
# kp1, des1 = orb.detectAndCompute(img1, None)


sift = cv.SIFT_create()
kp0 = sift.detect(img0, None)
kp1 = sift.detect(img1, None)

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp0, des0 = brief.compute(img0, kp0)
kp1, des1 = brief.compute(img1, kp1)


# orb = cv.ORB_create()
# kp0, des0 = orb.compute(img0, kp0)
# kp1, des1 = orb.compute(img1, kp1)
# 定义FLANN-based 匹配参数.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# 展示匹配
flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des0, des1, k=2)

matches = flann.knnMatch(np.float32(des0), np.float32(des1), k=2)
# 定义掩膜
mask_matches = [[0, 0] for i in range(len(matches))]

# 把mask_matches列表传递给cv2.drawMatchesKnn作为可选参数
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        mask_matches[i] = [1, 0]

# 绘制掩模中标记为好的匹配
# img_matches = cv.drawMatchesKnn(
#     img0, kp0, img1, kp1, matches, None,
#     matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
#     matchesMask=mask_matches, flags=0)
img_matches = cv.drawMatchesKnn(
    img0, kp0, img1, kp1, matches[:25], img1,
    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()
