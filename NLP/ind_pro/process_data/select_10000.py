# coding:utf-8

import json
import random

# ��������ԭʼ JSON ���ݴ洢��һ���ļ��У��ļ���Ϊ 'input_data.json'
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'
output_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_10000.json'

data = []
with open(input_file_path, 'r') as file:
    for line in file:
        item = json.loads(line)
        data.append(item)

# ��ȡ 10000 ����¼
if len(data) > 10000:
    extracted_data = random.sample(data, 10000)
else:
    extracted_data = data

# ����������ֶ�
extracted_data = [
    {
        'premise': item['premise'],
        'hypothesis': item['hypothesis'],
        'label': item['label']
    }
    for item in extracted_data
]

# ����ȡ�����ݱ��浽�µ� JSON �ļ���
with open(output_file_path, 'w') as file:
    json.dump(extracted_data, file, indent=4)

print(f"Extracted 10000 records and saved to {output_file_path}")