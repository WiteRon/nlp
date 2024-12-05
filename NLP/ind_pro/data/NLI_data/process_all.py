# coding:utf-8
import json
import os
import random

# ������������Ա�֤����ɸ��֣���ѡ��
random.seed(42)

# ָ��ԭʼ JSON �ļ���·��
input_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/all_data.json'  # �滻Ϊ���ԭʼ�ļ�·��

# ָ������ѵ��������֤����Ŀ¼
output_dir = '/home/zshiap/NLP/ind_pro/data/multi_nli_json'  # �滻Ϊ��ϣ�������Ŀ¼

# ȷ�����Ŀ¼����
os.makedirs(output_dir, exist_ok=True)

# ����ԭʼ JSON ����
with open(input_file_path, 'r') as f:
    data = json.load(f)

# �����������
random.shuffle(data)

# �� 80% �� 20% �ָ�����
train_size = int(0.8 * len(data))
train_data = data[:train_size]
validation_data = data[train_size:]

# �����µ�ѵ��������֤����ʽ
def create_new_format(dataset):
    new_dataset = []
    for entry in dataset:
        new_entry = {
            "input": f"premise: {entry['premise']}, hypothesis: {entry['hypothesis']}",
            "label": entry['label']
        }
        new_dataset.append(new_entry)
    return new_dataset

# ת�����ݸ�ʽ
train_data_formatted = create_new_format(train_data)
validation_data_formatted = create_new_format(validation_data)

# ����ѵ������ JSON �ļ�
with open(os.path.join(output_dir, "train_all.json"), "w") as train_file:
    json.dump(train_data_formatted, train_file, ensure_ascii=False, indent=4)

# ������֤���� JSON �ļ�
with open(os.path.join(output_dir, "validation_all.json"), "w") as validation_file:
    json.dump(validation_data_formatted, validation_file, ensure_ascii=False, indent=4)

print("jsons saved")