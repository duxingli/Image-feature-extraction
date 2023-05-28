"""

    cv2.findHomography() 是 OpenCV 中用于计算单应性矩阵（homography matrix）的函数。
    它可以从一组匹配的特征点中得到两幅图像之间的单应性变换关系。

    cv2.findHomography(src_points, dst_points, method, ransacReprojThreshold, mask) 函数有5个参数：

    src_points：
        源图像中的特征点坐标，是一个形如 (N, 1, 2) 的 numpy 数组，其中 N 是特征点数量。
    dst_points：
        目标图像中对应的特征点坐标，也是一个形如 (N, 1, 2) 的 numpy 数组，和 src_points 中的特征点一一对应。
    method：
        单应性矩阵的计算方法，默认为 0，表示使用所有点进行计算；如果设置为 cv2.RANSAC，则使用 RANSAC 算法寻找最优解。
    ransacReprojThreshold：
        RANSAC 算法中的阈值，表示计算点对之间距离时的最大容错距离，超过该距离的点对将被忽略。默认值为 3.0。
    mask：
        输出的掩码数组，用于标记哪些点对是内点（即符合单应性变换）以及哪些点对是外点（即不符合单应性变换）。如果不需要使用掩码则可以将其设为 None。
    函数返回两个值：

    H：计算得到的单应性矩阵，是一个 3x3 的变换矩阵。
    mask：输出的掩码数组，和输入参数中的 mask 数组一致，用于标记内点和外点。需要注意的是，如果 method 参数设置为 0，则该返回值将恒定为 None。
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 读入两张图片
img0 = cv.imread('./images/1.jpg')
img1 = cv.imread('./images/2.jpg')

# 转换为灰度图像
img0 = cv.cvtColor(img0,code=cv.COLOR_BGR2RGBA)
img1 = cv.cvtColor(img1,code=cv.COLOR_BGR2RGBA)
# img0 = cv.imread('./images/1.jpg', cv.IMREAD_GRAYSCALE)
# img1 = cv.imread('./images/2.jpg', cv.IMREAD_GRAYSCALE)
# 创建SIFT对象和特征点检测器
sift = cv.SIFT_create()
# 在两张图片中检测关键点和描述符
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# 使用FlannBased匹配算法进行特征点匹配
matcher = cv.FlannBasedMatcher()
matches = matcher.match(des0, des1)

# 筛选出好的匹配点对
good_matches = []
for m in matches:
    if m.distance < 0.5 * max([m.distance for m in matches]):
        good_matches.append(m)

# 获取匹配点对的坐标
src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法估计单应性矩阵
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=3.0)
H1, mask1 = cv.findHomography(dst_pts, src_pts, cv.RANSAC, ransacReprojThreshold=3.0)

w = img1.shape[0]
h = img1.shape[1]
rotate = cv.warpPerspective(img1, H1, (w, h),
borderMode=cv.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
plt.imshow(rotate)
plt.show()
