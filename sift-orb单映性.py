import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt

def addGaussNoise(image, mean, var):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def addSaltNoise(image, amount):  # 添加椒盐噪声函数
    output = image.copy()
    threshold = 1 - amount
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdm = random.random()
            if rdm < amount:
                output[i][j] = 0
            elif rdm > threshold:
                output[i][j] = 255
    return output

MIN_NUM_GOOD_MATCHES = 8

img0 = cv.imread('./images/book3.jpg', cv.IMREAD_GRAYSCALE)
img1 = cv.imread('./images/book4.jpg', cv.IMREAD_GRAYSCALE)

# img0 = addGaussNoise(img0, mean=0, var=0.001)
# img1 = addGaussNoise(img1, mean=0, var=0.001)
# img0 = addSaltNoise(img0, amount=0.005)
# img1 = addSaltNoise(img1, amount=0.005)

# sift = cv.SIFT_create()
# kp0, des0 = sift.detectAndCompute(img0, None)
# kp1, des1 = sift.detectAndCompute(img1, None)
#
sift = cv.SIFT_create()
kp0 = sift.detect(img0, None)
kp1 = sift.detect(img1, None)
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
kp0, des0 = brief.compute(img0, kp0)
kp1, des1 = brief.compute(img1, kp1)

# 定义 FLANN-based
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)


bf = cv.BFMatcher(cv.NORM_L1)
matches = bf.knnMatch(np.float32(des0), np.float32(des1),k=2)

# 展示 FLANN-based 匹配
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(np.float32(des0), np.float32(des1), k=2)
# 查询劳氏比率检验的匹配列表，
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
# 检查是否至少有10个好的匹配项
if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
    """
        如果满足这个条件，那么就查找匹配的关键点的二维坐标，
        并把这些坐标放入浮点坐标对的两个列表中。一个列表包含查询图像
        中的关键点坐标，另一个列表包含场景中匹配的关键点坐标
    """
    src_pts = np.float32(
        [kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 寻找单映性
    # 创建了mask_matches列表，将用于最终的匹配绘制，这样只有满足单应性的点才会绘制匹配线
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    M1, mask1 = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    mask_matches = mask.ravel().tolist()
    # 透视转换，取查询图像的矩形角点，并将其投影到场景中，这样就可以画出边界
    h, w = img0.shape
    src_corners = np.float32(
        [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst_corners = cv.perspectiveTransform(src_corners, M)
    dst_corners = dst_corners.astype(np.int32)

    # 依据单映性矩阵 画出匹配区域的边界
    num_corners = len(dst_corners)
    for i in range(num_corners):
        x0, y0 = dst_corners[i][0]
        if i == num_corners - 1:
            next_i = 0
        else:
            next_i = i + 1
        x1, y1 = dst_corners[next_i][0]
        cv.line(img1, (x0, y0), (x1, y1), 255, 3, cv.LINE_AA)

    # 画出通过 ratio test的数据.
    img_matches = cv.drawMatches(
        img0, kp0, img1, kp1, good_matches, None,
        matchColor=(0, 255, 0), singlePointColor=None,
        matchesMask=mask_matches, flags=2)

    # 显示最佳匹配
    plt.imshow(img_matches)
    plt.show()
else:
    print("Not enough matches good were found - %d/%d" % \
          (len(good_matches), MIN_NUM_GOOD_MATCHES))
img0 = cv.imread('./images/book3.jpg')
img1 = cv.imread('./images/book4.jpg')
# 转换为灰度图像
img0 = cv.cvtColor(img0,code=cv.COLOR_BGR2RGBA)
img1 = cv.cvtColor(img1,code=cv.COLOR_BGR2RGBA)
w = img1.shape[0]
h = img1.shape[1]
rotate = cv.warpPerspective(img1, M1, (w, h),
borderMode=cv.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
plt.imshow(rotate)
plt.show()