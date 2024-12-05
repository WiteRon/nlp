# coding:utf-8
import json
import random

# �������������ļ�·��
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json'  # �滻Ϊ���ԭʼ JSON �ļ�·��
train_output_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_train_5000.json'
val_output_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_val_1000.json'

# ����ԭʼ JSON ����
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ȷ���������㹻
if len(data) < 6000:
    raise ValueError("lack")

# �����ȡ 5000 ������ѵ����
train_data = random.sample(data, 5000)

# ��ԭʼ�����л�ȡʣ������ݣ�ȷ��û���ظ���Ŀ
remaining_data = [item for item in data if item not in train_data]  # ʹ���б��Ƶ�ʽ

# �����ȡ 1000 ��������֤��
val_data = random.sample(remaining_data, 1000)

# ����ѵ������ JSON �ļ�
with open(train_output_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

# ������֤���� JSON �ļ�
with open(val_output_path, 'w', encoding='utf-8') as val_file:
    json.dump(val_data, val_file, ensure_ascii=False, indent=4)

print("JSON saved")