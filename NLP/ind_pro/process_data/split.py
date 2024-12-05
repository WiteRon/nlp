# coding:utf-8

import json
import random

# 1. ��ȡ���� JSON �ļ�
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/multi_nli_10000.json'  # �����ļ�·��
output_train_file = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'  # ���ѵ�����ļ���
output_val_file = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/val.json'      # �����֤���ļ���

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. �������ݼ���80% ����ѵ����20% ������֤
random.seed(42)  # �������������ȷ�����ظ���
random.shuffle(data)  # ��������˳��

train_size = int(0.8 * len(data))
train_dataset = data[:train_size]
val_dataset = data[train_size:]

# 3. ����ѵ��������֤�����µ� JSON �ļ�
with open(output_train_file, 'w', encoding='utf-8') as train_file:
    json.dump(train_dataset, train_file, ensure_ascii=False, indent=4)

with open(output_val_file, 'w', encoding='utf-8') as val_file:
    json.dump(val_dataset, val_file, ensure_ascii=False, indent=4)

print("saved")