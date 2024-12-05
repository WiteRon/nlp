# coding:utf-8
from datasets import load_dataset
import json
import os
import pandas as pd
import numpy as np

# 加载数据集
ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

# 指定保存文件的路径
output_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/wiki.json'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 获取 evaluation 数据集部分并转换为 pandas DataFrame
df = ds['evaluation'].to_pandas()

# 将 DataFrame 转换为字典列表，并确保所有 ndarray 被转换为列表
data_list = df.to_dict(orient='records')

# 定义一个辅助函数来递归地将 numpy 对象转换为基本类型
def convert_numpy_objects(obj):
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_objects(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_objects(value) for key, value in obj.items()}
    return obj

# 使用辅助函数转换数据
data_list_converted = [convert_numpy_objects(entry) for entry in data_list]

# 保存为 JSON 格式，整个数据集作为一个数组保存
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(data_list_converted, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file_path}")