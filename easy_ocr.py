import easyocr
from PIL import Image, ImageEnhance, ImageOps

# 初始化 EasyOCR 识别器
reader = easyocr.Reader(['en'], gpu=False)  # 使用英语模型

# 图像处理
img = Image.open('./ruler拷貝.png')
# img = img.convert('L')
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(4)
# img = ImageOps.autocontrast(img)
# img = ImageOps.invert(img)

# 保存预处理后的图像并进行 OCR 识别
processed_image_path = './processed_image.png'
img.save(processed_image_path)

# 使用 EasyOCR 识别图像中的文本
results = reader.readtext(processed_image_path, detail=1)

# 过滤并显示识别到的数字
print("识别到的数字及其位置：")
for bbox, text, prob in results:
    if text.isdigit():  # 仅保留数字
        print(f"数字: {text}, 位置: {bbox}, 置信度: {prob}")
