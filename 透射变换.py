"""
需要确定原始图像中的四个点和目标图像中对应的四个点，在OpenCV中可以
    使用函数cv2.getPerspectiveTransform()来获得这些点之间的转换矩阵，
    并使用cv2.warpPerspective()函数将原始图像转换为目标图像

具体步骤如下：
    1. 确定原始图像中的四个点和目标图像中对应的四个点。
    2. 使用cv2.getPerspectiveTransform()函数获取转换矩阵。
    3. 使用cv2.warpPerspective()函数将原始图像转换为目标图像。
"""

import cv2
import numpy as np
import matplotlib as plt

# 读取原始图像
img = cv2.imread('./images/1.jpg')
# 原始图像中的四个点
pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])
# 目标图像中对应的四个点
pts2 = np.float32([[50,50],[100,50],[30,80],[80,80]])
# 获取透视变换矩阵
M = cv2.getPerspectiveTransform(pts1,pts2)

# shape = img.shape


# 将原始图像进行透视变换
dst = cv2.warpPerspective(img,M,(400,400),img, borderMode=cv2.BORDER_TRANSPARENT)

# 显示原始图像和目标图像
cv2.imshow('original',img)
cv2.imshow('perspective',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
