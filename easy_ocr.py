import easyocr
from PIL import Image, ImageEnhance
from collections import defaultdict

# 初始化 EasyOCR 處理器
reader = easyocr.Reader(['en'], gpu=False)  # 使用英語模型

# 讀取並處理圖像
img = Image.open('./ruler拷貝.png')
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(4)
processed_image_path = './processed_image.png'
img.save(processed_image_path)

# 使用 EasyOCR 识別圖像中的文本
results = reader.readtext(processed_image_path, detail=1)

# 過濾出檢測到的數字及其位置
detected_numbers = []
for bbox, text, prob in results:
    if text.isdigit():  # 僅保留數字
        x_avg = (bbox[0][0] + bbox[1][0]) / 2  # 計算 X 軸中心位置
        y_avg = (bbox[0][1] + bbox[2][1]) / 2  # 計算 Y 軸中心位置
        detected_numbers.append((int(text), x_avg, y_avg))
        print(f"已識別的數字: {text}, 位置: {bbox}, 置信度: {prob}")

# 按 X 軸位置分組相近的數字，假設同一尺標的數字 x 坐標相近
grouped_by_x = defaultdict(list)
threshold = 10  # X軸相似的阈值，可根據實際情況調整
for num, x, y in detected_numbers:
    found_group = False
    for gx in grouped_by_x:
        if abs(gx - x) < threshold:
            grouped_by_x[gx].append((num, x, y))
            found_group = True
            break
    if not found_group:
        grouped_by_x[x].append((num, x, y))

# 定義最終的數字位置字典，用來存儲 1 到 9 的位置
number_positions = {i: None for i in range(1, 10)}

# 排除不合理的元素，使剩下的元素能形成最長的有序序列
def find_longest_ordered_sequence(group):
    # 遞迴終止條件：若當前組已經形成有序序列（遞增或遞減）
    if all(group[i][0] < group[i + 1][0] for i in range(len(group) - 1)) or \
       all(group[i][0] > group[i + 1][0] for i in range(len(group) - 1)):
        return group  # 返回當前有序組

    # 記錄最大有序序列
    max_sequence = []
    
    # 嘗試排除每個元素並遞迴調用
    for i in range(len(group)):
        # 排除第 i 個元素
        temp_group = group[:i] + group[i + 1:]
        
        # 遞迴調用，繼續尋找有序序列
        ordered_sequence = find_longest_ordered_sequence(temp_group)
        
        # 更新最大序列
        if len(ordered_sequence) > len(max_sequence):
            max_sequence = ordered_sequence
            
    return max_sequence

# 輸出分組後的結果
print("分組後的結果：", grouped_by_x)

# 遍歷每個分組並過濾不合理數字
for group in grouped_by_x.values():
    # 按 Y 軸位置排序
    group.sort(key=lambda item: item[2])
    # 找到最長的有序序列
    filtered_group = find_longest_ordered_sequence(group)

    # 填入合理的數字位置
    for num, x, y in filtered_group:
        if num >= 1 and num <= 9:
            number_positions[num] = (x, y)

print("初始位置字典：", number_positions)

# 提取已存在的數字和它們的 X, Y 坐標
existing_numbers = sorted([(num, pos[0], pos[1]) for num, pos in number_positions.items() if pos is not None])

print("已知數字及其 X 和 Y 軸位置：", existing_numbers)

# 計算已知數字之間的單位 X, Y 差異
unit_diffs = {}  # 存儲每個數字與其他數字的單位差異
for i, (num1, x1, y1) in enumerate(existing_numbers):
    x_diffs, y_diffs = [], []
    for j, (num2, x2, y2) in enumerate(existing_numbers):
        if i != j:
            # 計算單位 X 和 Y 差異
            unit_x_diff = (x2 - x1) / (num2 - num1)
            unit_y_diff = (y2 - y1) / (num2 - num1)
            x_diffs.append(unit_x_diff)
            y_diffs.append(unit_y_diff)
    # 將該數字與其他數字的單位差異存儲
    unit_diffs[num1] = (x_diffs, y_diffs)

# 計算每個缺失數字的位置
for i in range(1, 10):
    if number_positions[i] is None:
        # 計算與每個已知數字的單位差異加權估算
        estimated_xs, estimated_ys = [], []
        for known_num, known_x, known_y in existing_numbers:
            if known_num in unit_diffs and unit_diffs[known_num][0]:  # 確保單位差異非空
                avg_x_diff = sum(unit_diffs[known_num][0]) / len(unit_diffs[known_num][0])
                avg_y_diff = sum(unit_diffs[known_num][1]) / len(unit_diffs[known_num][1])
                
                # 計算基於該已知數字的估算 X, Y 值
                estimated_x = known_x + (i - known_num) * avg_x_diff
                estimated_y = known_y + (i - known_num) * avg_y_diff
                estimated_xs.append(estimated_x)
                estimated_ys.append(estimated_y)
        
        # 對所有估算的 X 和 Y 值求平均，作為最終位置
        final_estimated_x = sum(estimated_xs) / len(estimated_xs) if estimated_xs else existing_numbers[0][1]
        final_estimated_y = sum(estimated_ys) / len(estimated_ys) if estimated_ys else existing_numbers[0][2]
        number_positions[i] = (round(final_estimated_x,2), round(final_estimated_y,2))

# 輸出最終的數字位置字典
print("完整的尺標數字及其 X, Y 軸位置：")
for num in range(1, 10):
    if number_positions[num] is not None:
        x, y = number_positions[num]
        print(f"數字: {num}, X軸位置: {x}, Y軸位置: {y}")
    else:
        print(f"數字: {num} 無法推算出位置")
