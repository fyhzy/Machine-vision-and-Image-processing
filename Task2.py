import cv2
import numpy as np


# 对轮廓列表进行排序
def sort_contours(cnts, method="left-to-right"):
    # 表示不需要反向排序
    reverse = False
    # 当 i 的值为 0 时，表示在排序时使用 x 轴的坐标值
    i = 0

    # 如果 method 参数的值是 "right-to-left" 或 "bottom-to-top" 则 reverse 被设置为 True 表示进行反向排序
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # method 参数的值是 "top-to-bottom" 或 "bottom-to-top"，则 i 被设置为 1，表示在排序时使用 y 轴的坐标值
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 用一个最小的矩形，把找到的形状包起来，用x、y、h、w表示
    # cv2.boundingRect(c) 返回一个包含四个值的元组 (x, y, w, h) 其中(x, y)是外接矩形的左上角坐标 w 是矩形的宽度，h 是矩形的高度
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # zip函数用于打包可迭代数据，得到最终输出的 cnts 和 boundingBoxes
    # 将 cnts 列表和 boundingBoxes 列表中的元素一一配对
    # 每个配对都是包含两个元素的元组 第一个元素是一个轮廓 第二个元素是对应轮廓的外接矩形信息
    # 按照元组中的第二个元素的第 i 个坐标值进行排序
    # * 操作符用于解包 将排序后的元组列表拆分成两个独立的列表 即 cnts 和 boundingBoxes
    # 使用 zip(*...) 将排序后的元组列表拆分为两个独立的列表 即排序后的 轮廓列表 cnts 和外接矩形列表 boundingBoxes
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


# 调整图像大小的函数 根据指定的宽度和高度来调整图像的尺寸
# height 为None 则根据宽度进行等比例调整
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    # 返回一个包含图像高度、宽度和通道数的元组
    # 通过使用切片操作 [:2] 我们可以从这个元组中提取前两个元素 即图像的高度和宽度
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # 对输入图像 image 进行大小调整，并将结果保存在 resized 变量中
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 读取模板图像
template = cv2.imread('images/testim-4.png')

# 将模板图像转换为灰度图
image_Gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

# 将灰度图中的像素值与指定的阈值进行比较 将大于阈值的像素置为白色（255）小于等于阈值的像素置为黑色（0）
# 得到一个二值图像
image_Binary = cv2.threshold(image_Gray, 177, 255, cv2.THRESH_BINARY_INV)[1]
# 使用 cv2.findContours 函数查找二值图像中的轮廓
# cv2.RETR_EXTERNAL 表示只检测外部轮廓 cv2.CHAIN_APPROX_SIMPLE 表示轮廓的近似方法
# findContours函数返回值 0-轮廓列表 1-层次结构
refcnts = cv2.findContours(image_Binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
refcnts = sort_contours(refcnts, method="left-to-right")[0]
digits = {}

# 遍历每个轮廓
# 使用 enumerate 函数遍历 refcnts 列表中的每个轮廓 获取索引 i 和轮廓 c
for (i, c) in enumerate(refcnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 从二值化图像中提取了一个数字矩形区域 并将其保存到roi变量中
    roi = image_Binary[y:y + h, x:x + w]
    # 将roi变量中的数字矩形区域缩放成了指定大小(58, 88)的矩形
    roi = cv2.resize(roi, (58, 88))
    digits[i] = roi

# 创建形态学操作的结构元素
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取图像并调整大小
image = cv2.imread("images/testim-3.png")
image = resize(image, width=300)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用形态学操作进行图像处理 突出图像中的一些特定区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# 使用Sobel算子进行梯度计算
gradx = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradx = np.absolute(gradx)

# 归一化梯度值到0-255
minVal = np.min(gradx)
maxVal = np.max(gradx)
gradx = (255 * ((gradx - minVal) / (maxVal - minVal)))
gradx = gradx.astype("uint8")

# 闭运算，将图像中的小孔或小黑点进行填充
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectKernel)

# 二值化处理，使用OTSU自适应阈值
thresh = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 查找二值图像中的外部轮廓
threshCnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cnts = threshCnts
locs = []

# 使用enumerate遍历cnts列表中的轮廓 返回索引i和轮廓c
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 筛选出宽高比在2.5到5.0之间 宽度在40到85之间 高度在10到20之间的轮廓 将这些符合条件的轮廓信息添加到locs列表中
    if 2.5 < ar < 5.0 and 40 < w < 80 and 10 < h < 20:
        locs.append((x, y, w, h))

# 对locs列表中的轮廓按照横坐标x进行排序
locs = sorted(locs, key=lambda x: x[0])

output = []
# 遍历轮廓中的每一个数字
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    # 初始化一个空列表 groupOutput，用于存储当前数字区域中每个数字的识别结果
    groupOutput = []
    # 提取数字区域的灰度图像，并在上 下 左 右各扩展5个像素的边界范围
    group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]

    # 对提取的数字区域进行二值化处理 使用OTSU自适应阈值
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 使用 cv2.findContours 函数找到二值图像中的轮廓，这些轮廓应该对应于数字的各个部分
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 使用 sort_contours 函数对找到的轮廓按照从左到右的顺序进行排序
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 使用 cv2.boundingRect 获取每个数字轮廓的外接矩形 (x, y, w, h)
        (x, y, w, h) = cv2.boundingRect(c)

        # 提取数字区域中的每个数字 存放在 roi 变量中
        roi = group[y:y + h, x:x + w]
        # 调整为固定的大小 (58, 88)
        roi = cv2.resize(roi, (58, 88))

        # 初始化一个空列表 scores 用于存储当前数字与模板数字的匹配分数
        scores = []

        # 遍历预先准备好的数字模板 digits 中的每个数字
        for (digit, digitROI) in digits.items():
            # 使用 cv2.matchTemplate 函数计算模板匹配的结果
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF_NORMED)

            # 使用 cv2.minMaxLoc 获取匹配结果的最大值 即匹配分数
            score = cv2.minMaxLoc(result)[1]

            # 将匹配分数添加到 scores 列表中
            scores.append(score)

        # 使用 np.argmax(scores) 找到 scores 列表中分数最高的索引 即对应的数字
        # 将最高分数的数字添加到 groupOutput 列表中
        groupOutput.append(str(np.argmax(scores)))

    # 将 groupOutput 列表中的数字添加到 output 列表中
    output.extend(groupOutput)

# 画出矩形和文本
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    # 通过索引操作 output[i * 4:(i + 1) * 4] 从 output 中选择了与当前数字区域相关的四个数字的识别结果
    # 然后使用 "".join(...) 将这四个数字的识别结果拼接成一个字符串
    groupOutput = "".join(output[i * 4:(i + 1) * 4])

    # 画矩形框 标记数字区域的位置
    cv2.rectangle(image, (gx, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 2)

    # 添加文本 标记数字区域的识别结果
    cv2.putText(image, groupOutput, (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

print(output)
# 保存和显示最终结果
cv2.imwrite('result2.png', image)
cv2.imshow('result2', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
