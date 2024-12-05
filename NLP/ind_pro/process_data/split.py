# coding:utf-8

import json
import random

# 1. 读取本地 JSON 文件
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/multi_nli_10000.json'  # 输入文件路径
output_train_file = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'  # 输出训练集文件名
output_val_file = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/val.json'      # 输出验证集文件名

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 划分数据集，80% 用于训练，20% 用于验证
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(data)  # 打乱数据顺序

train_size = int(0.8 * len(data))
train_dataset = data[:train_size]
val_dataset = data[train_size:]

# 3. 保存训练集和验证集到新的 JSON 文件
with open(output_train_file, 'w', encoding='utf-8') as train_file:
    json.dump(train_dataset, train_file, ensure_ascii=False, indent=4)

with open(output_val_file, 'w', encoding='utf-8') as val_file:
    json.dump(val_dataset, val_file, ensure_ascii=False, indent=4)

print("saved")