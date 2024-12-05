# coding:utf-8
import json
import os
import random

# 设置随机种子以保证结果可复现（可选）
random.seed(42)

# 指定原始 JSON 文件的路径
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/all_data.json'  # 替换为你的原始文件路径

# 指定保存训练集和验证集的目录
output_dir = '/home/zshiap/NLP/ind_pro/data/multi_nli_json'  # 替换为你希望保存的目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载原始 JSON 数据
with open(input_file_path, 'r') as f:
    data = json.load(f)

# 随机打乱数据
random.shuffle(data)

# 按 80% 和 20% 分割数据
train_size = int(0.8 * len(data))
train_data = data[:train_size]
validation_data = data[train_size:]

# 创建新的训练集和验证集格式
def create_new_format(dataset):
    new_dataset = []
    for entry in dataset:
        new_entry = {
            "input": f"premise: {entry['premise']}, hypothesis: {entry['hypothesis']}",
            "label": entry['label']
        }
        new_dataset.append(new_entry)
    return new_dataset

# 转换数据格式
train_data_formatted = create_new_format(train_data)
validation_data_formatted = create_new_format(validation_data)

# 保存训练集到 JSON 文件
with open(os.path.join(output_dir, "train_all.json"), "w") as train_file:
    json.dump(train_data_formatted, train_file, ensure_ascii=False, indent=4)

# 保存验证集到 JSON 文件
with open(os.path.join(output_dir, "validation_all.json"), "w") as validation_file:
    json.dump(validation_data_formatted, validation_file, ensure_ascii=False, indent=4)

print("jsons saved")