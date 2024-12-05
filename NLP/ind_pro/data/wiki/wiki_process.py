# coding:utf-8
import json
import os
from collections import Counter
import random

# ������������Ա�֤����ɸ��֣������Ҫ�Ļ���
random.seed(42)

# ָ��ԭʼ JSON �ļ���·��
input_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/wiki.json'  # �滻Ϊ���ԭʼ�ļ�·��

# ָ�������¸�ʽ���ݵ�Ŀ¼
output_dir = '/home/zshiap/NLP/ind_pro/data/wiki'  # �滻Ϊ��ϣ�������Ŀ¼

# ȷ�����Ŀ¼����
os.makedirs(output_dir, exist_ok=True)

# ����ԭʼ JSON ����
with open(input_file_path, 'r') as f:
    data = json.load(f)

# ������ǩ�����ֵ�ӳ��
label_mapping = {
    "major_inaccurate": 2,
    "accurate": 0,
    "minor_inaccurate": 1
}

# �����µ����ݸ�ʽ
def create_new_format(dataset):
    new_dataset = []
    for entry in dataset:
        # ������Ƶ�����ֵı�ע
        label_counts = Counter(entry['annotation'])
        most_common_label = label_counts.most_common(1)[0][0]

        # ʹ��ӳ��ת����ǩ
        mapped_label = label_mapping.get(most_common_label, most_common_label)  # �����ǩ����ӳ���У��򱣳�ԭ��

        # �����µ������ʽ
        new_entry = {
            "input": f"premise: {entry['wiki_bio_text']}, hypothesis: {', '.join(entry['gpt3_sentences'])}",
            "label": mapped_label
        }
        new_dataset.append(new_entry)
    return new_dataset

# ת�����ݸ�ʽ
formatted_data = create_new_format(data)

# ����ת������¸�ʽ���ݵ� JSON �ļ�
output_file_path = os.path.join(output_dir, "formatted_data.json")
with open(output_file_path, "w") as output_file:
    json.dump(formatted_data, output_file, ensure_ascii=False, indent=4)

print("json saved")