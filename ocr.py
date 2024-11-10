from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# 初始化 OCR 模型
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 'ch' 表示支援中文，如果只需要英文可使用 'en'

# 讀取影像
img = Image.open('./ruler拷貝.png')

# 將圖像轉為灰階模式便於處理
img = img.convert('L')

# 將圖像二值化：接近黑色的像素保留為黑色，其餘轉為白色
threshold = 50  # 調整閾值以控制「黑色」的範圍
img = img.point(lambda p: 0 if p < threshold else 255)

# 將圖像轉換回灰度或 RGB 模式
img = img.convert('L')  # 或者使用 img = img.convert('RGB')

# 增強對比度
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(4)

# 保存處理後的圖像，便於檢查
processed_image_path = './processed_image.png'
img.save(processed_image_path)

# 使用 PaddleOCR 進行文字識別
result = ocr.ocr(np.array(img), cls=True)

# 儲存檢測到的數字及其位置
digits_with_position = []

# 解析 OCR 輸出結果
for line in result[0]:
    text = line[1][0]  # 識別的文本內容
    if text.isdigit():  # 僅處理數字
        x1, y1 = line[0][0]  # 左上角坐標
        x2, y2 = line[0][2]  # 右下角坐標
        x_avg = (x1 + x2) / 2  # X軸中心位置
        y_avg = (y1 + y2) / 2  # Y軸中心位置
        digits_with_position.append((text, x_avg, y_avg))
        print(f"數字: {text}, 位置: {line[0]}")

# 按 X 軸位置排序
digits_with_position.sort(key=lambda x: x[1])

# 顯示提取的數字及其對應的 Y 軸位置
print("提取的尺標數字及其Y軸位置：")
for digit, x, y in digits_with_position:
    print(f"數字: {digit}, Y軸位置: {y}")
