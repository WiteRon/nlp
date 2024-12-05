# coding:utf-8
from datasets import load_dataset
import json
import os
import pandas as pd
import numpy as np

# �������ݼ�
ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

# ָ�������ļ���·��
output_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/wiki.json'

# ȷ�����Ŀ¼����
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# ��ȡ evaluation ���ݼ����ֲ�ת��Ϊ pandas DataFrame
df = ds['evaluation'].to_pandas()

# �� DataFrame ת��Ϊ�ֵ��б���ȷ������ ndarray ��ת��Ϊ�б�
data_list = df.to_dict(orient='records')

# ����һ�������������ݹ�ؽ� numpy ����ת��Ϊ��������
def convert_numpy_objects(obj):
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_objects(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_objects(value) for key, value in obj.items()}
    return obj

# ʹ�ø�������ת������
data_list_converted = [convert_numpy_objects(entry) for entry in data_list]

# ����Ϊ JSON ��ʽ���������ݼ���Ϊһ�����鱣��
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(data_list_converted, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file_path}")