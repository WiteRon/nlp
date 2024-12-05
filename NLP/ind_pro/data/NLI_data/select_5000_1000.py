# coding:utf-8
import json
import random

# 定义输入和输出文件路径
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'  # 替换为你的原始 JSON 文件路径
train_output_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_train_5000.json'
val_output_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_val_1000.json'

# 加载原始 JSON 数据
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 确保数据量足够
if len(data) < 6000:
    raise ValueError("lack")

# 随机抽取 5000 条用于训练集
train_data = random.sample(data, 5000)

# 从原始数据中获取剩余的数据，确保没有重复条目
remaining_data = [item for item in data if item not in train_data]  # 使用列表推导式

# 随机抽取 1000 条用于验证集
val_data = random.sample(remaining_data, 1000)

# 保存训练集到 JSON 文件
with open(train_output_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

# 保存验证集到 JSON 文件
with open(val_output_path, 'w', encoding='utf-8') as val_file:
    json.dump(val_data, val_file, ensure_ascii=False, indent=4)

print("JSON saved")