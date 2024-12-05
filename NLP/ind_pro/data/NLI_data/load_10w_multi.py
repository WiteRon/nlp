# coding:utf-8
from datasets import load_dataset
import json
import os

# 指定下载的目录
download_dir = "/home/zshiap/NLP/ind_pro/data/multi_nli_json"  # 替换为你希望保存的目录

# 确保目录存在
os.makedirs(download_dir, exist_ok=True)

# 加载数据集
dataset = load_dataset("nyu-mll/multi_nli")

# 将训练集和验证集保存为 JSON 文件
train_data = dataset['train']
validation_matched_data = dataset['validation_matched']
validation_mismatched_data = dataset['validation_mismatched']

# 保存训练集
with open(os.path.join(download_dir, "multi_nli_train.json"), "w") as f:
    json.dump(train_data.to_list(), f)  # 转换为列表，每个元素是一个字典

# 保存验证集（匹配）
with open(os.path.join(download_dir, "multi_nli_validation_matched.json"), "w") as f:
    json.dump(validation_matched_data.to_list(), f)  # 转换为列表

# 保存验证集（不匹配）
with open(os.path.join(download_dir, "multi_nli_validation_mismatched.json"), "w") as f:
    json.dump(validation_mismatched_data.to_list(), f)  # 转换为列表

print("JSON saved")