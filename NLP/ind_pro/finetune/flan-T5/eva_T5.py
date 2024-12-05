# coding:utf-8

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ����΢�����ģ�ͺ� tokenizer
model_path = '/home/zshiap/NLP/ind_pro/finetune/models/T5/model'  # ȷ��·����ȷ
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# ����п��õ�GPU����ʹ��GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ����ģ��Ϊ����ģʽ
model.eval()

# ����������
def infer(input_string):
    # ��������б��룬������ע��������
    inputs = tokenizer(input_string, return_tensors='pt', padding=True, truncation=True, max_length=256)
    
    # �������Ƶ�ָ���豸��
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # ʹ��ģ�ͽ���Ԥ��
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities).item()

    return prediction, confidence, probabilities.tolist()[0]

# ���ز�������
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# �����򲿷֣���������ָ��
if __name__ == "__main__":
    test_file_path = '/home/zshiap/NLP/ind_pro/data/wiki/formatted_data.json'  # ָ�����������ļ�·��
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

    # ��������ָ��
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    # �������ָ��
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")