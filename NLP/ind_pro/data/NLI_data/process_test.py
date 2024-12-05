# coding:utf-8
import json

# ���庯����ת��������Ŀ
def transform_entry(entry):
    # ����һ���µ��ֵ��Ա���ת���������
    transformed = {}
    
    # ��װ "input" �ֶΣ�����ָ����ʽ
    transformed["input"] = f"premise: {entry['sentence1']}, hypothesis: {entry['sentence2']}"
    
    # �� "gold_label" ת��Ϊ���ֱ�ǩ��������� 'contradiction'=2, 'neutral'=1, 'entailment'=0��
    label_mapping = {'contradiction': 2, 'neutral': 1, 'entailment': 0}
    transformed["label"] = label_mapping.get(entry['gold_label'], -1)  # ���û��ƥ�����Ĭ��ֵΪ-1
    
    return transformed

# ����������һ����Ϊ input.jsonl ���ļ�����ԭʼ����
input_file_path = '/home/zshiap/start/sftp/dev_mismatched_sampled-1.jsonl'
output_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/dismatch.json'

# �����������ļ�
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
     
    # ��ʼ��һ���б����������е�ת�������Ŀ
    transformed_entries = []
    
    # ���ж�ȡ��ת��ÿһ�� JSON ����
    for line in infile:
        entry = json.loads(line.strip())
        transformed_entries.append(transform_entry(entry))
    
    # ������ת�������Ŀ��Ϊ JSON ����д�뵽����ļ���
    json.dump(transformed_entries, outfile, ensure_ascii=False, indent=2)