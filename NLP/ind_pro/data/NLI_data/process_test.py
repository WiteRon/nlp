# coding:utf-8
import json

# 定义函数来转换单个条目
def transform_entry(entry):
    # 创建一个新的字典以保存转换后的数据
    transformed = {}
    
    # 组装 "input" 字段，按照指定格式
    transformed["input"] = f"premise: {entry['sentence1']}, hypothesis: {entry['sentence2']}"
    
    # 将 "gold_label" 转换为数字标签（这里假设 'contradiction'=2, 'neutral'=1, 'entailment'=0）
    label_mapping = {'contradiction': 2, 'neutral': 1, 'entailment': 0}
    transformed["label"] = label_mapping.get(entry['gold_label'], -1)  # 如果没有匹配项，则默认值为-1
    
    return transformed

# 假设我们有一个名为 input.jsonl 的文件包含原始数据
input_file_path = '/home/zshiap/start/sftp/dev_mismatched_sampled-1.jsonl'
output_file_path = '/home/zshiap/NLP/ind_pro/data/multi_nli_json/dismatch.json'

# 打开输入和输出文件
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
     
    # 初始化一个列表来保存所有的转换后的条目
    transformed_entries = []
    
    # 逐行读取并转换每一个 JSON 对象
    for line in infile:
        entry = json.loads(line.strip())
        transformed_entries.append(transform_entry(entry))
    
    # 将所有转换后的条目作为 JSON 数组写入到输出文件中
    json.dump(transformed_entries, outfile, ensure_ascii=False, indent=2)