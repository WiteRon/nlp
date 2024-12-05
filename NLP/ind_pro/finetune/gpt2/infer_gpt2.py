# coding:utf-8

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# 加载微调后的模型和 tokenizer
model_path = '/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model'  # 确保路径正确
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

# 主程序部分，使用预定义的提示词
if __name__ == "__main__":
    input_string = "premise: Only Helms went out of his way this time., hypothesis: The others did not have the courage to go out of their way, like Helms."
      
      
       
    try:
        prediction, confidence, probabilities = infer(input_string)

        # 输出预测结果
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        print(f"Prediction: {label_map[prediction]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probabilities}")

    except Exception as e:
        print(f"Error: {e}")