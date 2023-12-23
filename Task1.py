import cv2


# 通过自适应直方图均衡化来提高图像的对比度
# 使用高斯模糊进行降噪
def enhance_contrast(img):
    # 使用高斯模糊进行降噪
    img = cv2.GaussianBlur(image, (3, 3), 0)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 将增强对比度的灰度图转回彩色图
    enhanced_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_img


# 读取图片
image = cv2.imread('images/testim-1.jpg')

enhanced_image = enhance_contrast(image)

# 函数用于将图像从BGR颜色空间（默认情况下OpenCV读取的图像是BGR格式）转换为灰度图
gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

# 将灰度图中的像素值与指定的阈值进行比较 将大于阈值的像素置为白色（255）
# 小于等于阈值的像素置为黑色（0） 这样就得到了一个二值图像 其中米粒的部分是白色的
# thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# 使用自适应阈值处理
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 1)

# 寻找图像中的轮廓 它返回一个包含所有轮廓的列表
# cv2.RETR_EXTERNAL参数表示只检测外部轮廓 cv2.CHAIN_APPROX_SIMPLE参数表示仅保留轮廓的端点 以节省内存
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# 以写文件权限打开文本文件，准备写入坐标信息
with open('result1.txt', 'w') as file:
    # 遍历每个轮廓，计算包围框和面积
    total_rice = 0
    # enumerate(contours) 返回一个轮廓列表 contours 中每个元素的索引 i 和相应的轮廓 contour
    for i, contour in enumerate(contours):
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 过滤小面积的轮廓
        if area > 300:
            total_rice += 1
            # 计算轮廓的包围矩形 返回一个矩形
            # 其左上角坐标是 (x, y) 宽度是 w 高度是 h 这个矩形是能够完全包含输入轮廓的最小矩形。
            x, y, w, h = cv2.boundingRect(contour)
            # 输出每个米粒的坐标信息到文件 矩形的四个顶点坐标
            rect_info = f"Rice {total_rice}: ({x}, {y}), ({x + w}, {y}), ({x + w}, {y + h}), ({x}, {y + h})\n"
            file.write(rect_info)
            # 画出矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 输出总米粒数
print(f"米粒总数有：{total_rice} 个")

# 将画了矩形框的结果图保存
cv2.imwrite('result1.jpg', image)
cv2.imshow('result1', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
