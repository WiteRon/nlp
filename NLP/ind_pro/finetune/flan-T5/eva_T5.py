# coding:utf-8

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 加载微调后的模型和 tokenizer
model_path = '/home/zshiap/NLP/ind_pro/finetune/models/T5/model'  # 确保路径正确
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# 如果有可用的GPU，则使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置模型为评估模式
model.eval()

# 定义推理函数
def infer(input_string):
    # 对输入进行编码，并生成注意力掩码
    inputs = tokenizer(input_string, return_tensors='pt', padding=True, truncation=True, max_length=256)
    
    # 将输入移到指定设备上
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities).item()

    return prediction, confidence, probabilities.tolist()[0]

# 加载测试数据
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 主程序部分，计算评估指标
if __name__ == "__main__":
    test_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/formatted_data.json'  # 指定测试数据文件路径
    test_data = load_test_data(test_file_path)

    true_labels = []
    predictions = []

    for item in test_data:
        input_string = item['input']
        label = item['label']

        try:
            prediction, _, _ = infer(input_string)
            true_labels.append(label)
            predictions.append(prediction)

        except Exception as e:
            print(f"Error processing input: {input_string}, Error: {e}")

    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    # 输出评估指标
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")