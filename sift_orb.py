import cv2

# 加载图像
img = cv2.imread('./images/1.jpg')

# 初始化 SIFT 和 ORB 特征提取器
sift = cv2.SIFT_create(1)
orb = cv2.ORB_create()

# 检测 SIFT 关键点

# sift.setContrastThreshold(0.01)

kp = sift.detect(img, None)
# 计算 SIFT 描述子
kp, des = sift.compute(img, kp)

# 用 SIFT 关键点和描述子计算 ORB 描述子
kp_orb, des_orb = orb.compute(img, kp)

# 显示结果
img = cv2.drawKeypoints(img, kp_orb, None)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
