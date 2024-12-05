# coding:utf-8
from datasets import load_dataset
import json
import os

# ָ�����ص�Ŀ¼
download_dir = "/home/zshiap/NLP/ind_pro/data/multi_nli_json"  # �滻Ϊ��ϣ�������Ŀ¼

# ȷ��Ŀ¼����
os.makedirs(download_dir, exist_ok=True)

# �������ݼ�
dataset = load_dataset("nyu-mll/multi_nli")

# ��ѵ��������֤������Ϊ JSON �ļ�
train_data = dataset['train']
validation_matched_data = dataset['validation_matched']
validation_mismatched_data = dataset['validation_mismatched']

# ����ѵ����
with open(os.path.join(download_dir, "multi_nli_train.json"), "w") as f:
    json.dump(train_data.to_list(), f)  # ת��Ϊ�б�ÿ��Ԫ����һ���ֵ�

# ������֤����ƥ�䣩
with open(os.path.join(download_dir, "multi_nli_validation_matched.json"), "w") as f:
    json.dump(validation_matched_data.to_list(), f)  # ת��Ϊ�б�

# ������֤������ƥ�䣩
with open(os.path.join(download_dir, "multi_nli_validation_mismatched.json"), "w") as f:
    json.dump(validation_mismatched_data.to_list(), f)  # ת��Ϊ�б�

print("JSON saved")