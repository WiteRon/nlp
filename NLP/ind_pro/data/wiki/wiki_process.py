# coding:utf-8
import json
import os
from collections import Counter
import random

# 设置随机种子以保证结果可复现（如果需要的话）
random.seed(42)

# 指定原始 JSON 文件的路径
input_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/wiki.json'  # 替换为你的原始文件路径

# 指定保存新格式数据的目录
output_dir = '/home/zshiap/NLP/ind_pro/data/wiki'  # 替换为你希望保存的目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载原始 JSON 数据
with open(input_file_path, 'r') as f:
    data = json.load(f)

# 创建标签到数字的映射
label_mapping = {
    "major_inaccurate": 2,
    "accurate": 0,
    "minor_inaccurate": 1
}

# 创建新的数据格式
def create_new_format(dataset):
    new_dataset = []
    for entry in dataset:
        # 计算最频繁出现的标注
        label_counts = Counter(entry['annotation'])
        most_common_label = label_counts.most_common(1)[0][0]

        # 使用映射转换标签
        mapped_label = label_mapping.get(most_common_label, most_common_label)  # 如果标签不在映射中，则保持原样

        # 构造新的输入格式
        new_entry = {
            "input": f"premise: {entry['wiki_bio_text']}, hypothesis: {', '.join(entry['gpt3_sentences'])}",
            "label": mapped_label
        }
        new_dataset.append(new_entry)
    return new_dataset

# 转换数据格式
formatted_data = create_new_format(data)

# 保存转换后的新格式数据到 JSON 文件
output_file_path = os.path.join(output_dir, "formatted_data.json")
with open(output_file_path, "w") as output_file:
    json.dump(formatted_data, output_file, ensure_ascii=False, indent=4)

print("json saved")