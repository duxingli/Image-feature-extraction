# SURF算法  检测尺度不变换 ————————用不了

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 读取图像
img = cv.imread('./images/1.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 使用 sift关键点检测
# 实例化sift对象
surf = cv.xfeatures2d.SURF_create(8000)
# sift = cv.SIFT_create()
# 关键点检测：kp关键点信息包括方向，尺度，位置信息，des是关键点的描述符
kp,des= surf.detectAndCompute(gray,None)
# 在图像上绘制关键点的检测结果
cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 图像显示
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(img[:,:,::-1])
plt.title('sift检测')
plt.xticks([]), plt.yticks([])
plt.show()




