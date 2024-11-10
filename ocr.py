from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageEnhance
import numpy as np

# 初始化 OCR 模型
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 'ch' 表示支援中文，如果只需要英文可使用 'en'

# 讀取並處理影像
img = Image.open('./ruler拷貝.png')
# img = img.convert('L')
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(4)
# img = ImageOps.autocontrast(img)
# img = ImageOps.invert(img)

# 保存预处理后的图像并进行 OCR 识别
processed_image_path = './processed_image.png'
img.save(processed_image_path)
# 將 PIL Image 轉換為 numpy array 格式
# img_np = np.array(img)

digits_with_position = []

# 使用 PaddleOCR 進行文字識別
result = ocr.ocr(processed_image_path, cls=True)

print(result)

for line in result[0]:
    text = line[1][0]  # 识别的文本
    if text.isdigit():  # 检查是否为数字
        x1, y1 = line[0][0]  # 左上角坐标
        x2, y2 = line[0][2]  # 右下角坐标
        x_avg = (x1 + x2) / 2  # X轴中心位置
        y_avg = (y1 + y2) / 2  # Y轴中心位置
        digits_with_position.append((text, x_avg, y_avg))
    print(f"数字: {text}, 位置: {line[0]}")

# 按 X 轴位置排序
digits_with_position.sort(key=lambda x: x[1])

# 显示提取的数字和Y轴位置
print("提取的尺标数字及其Y轴位置：")
for digit, x, y in digits_with_position:
    print(f"数字: {digit}, Y轴位置: {y}")
