# coding:utf-8

import json
import random

# 假设您的原始 JSON 数据存储在一个文件中，文件名为 'input_data.json'
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'
output_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_10000.json'

data = []
with open(input_file_path, 'r') as file:
    for line in file:
        item = json.loads(line)
        data.append(item)

# 抽取 10000 条记录
if len(data) > 10000:
    extracted_data = random.sample(data, 10000)
else:
    extracted_data = data

# 保留所需的字段
extracted_data = [
    {
        'premise': item['premise'],
        'hypothesis': item['hypothesis'],
        'label': item['label']
    }
    for item in extracted_data
]

# 将抽取的数据保存到新的 JSON 文件中
with open(output_file_path, 'w') as file:
    json.dump(extracted_data, file, indent=4)

print(f"Extracted 10000 records and saved to {output_file_path}")