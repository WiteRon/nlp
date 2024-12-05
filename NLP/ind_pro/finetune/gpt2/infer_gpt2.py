# coding:utf-8

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# ����΢�����ģ�ͺ� tokenizer
model_path = '/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model'  # ȷ��·����ȷ
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

# �����򲿷֣�ʹ��Ԥ�������ʾ��
if __name__ == "__main__":
    input_string = "premise: Only Helms went out of his way this time., hypothesis: The others did not have the courage to go out of their way, like Helms."
      
      
       
    try:
        prediction, confidence, probabilities = infer(input_string)

        # ���Ԥ����
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        print(f"Prediction: {label_map[prediction]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probabilities}")

    except Exception as e:
        print(f"Error: {e}")