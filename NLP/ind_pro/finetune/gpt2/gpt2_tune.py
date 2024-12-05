# coding:utf-8

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# 加载数据集
dataset = load_dataset('json', data_files={
    'train': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json',
    'validation': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/validation.json'
})

download_and_save_dir = "/home/zshiap/NLP/ind_pro/data/llms/gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=download_and_save_dir)
# 设置填充标记为结束标记
tokenizer.pad_token = tokenizer.eos_token

# 数据预处理
def tokenize(batch):
    tokenized_inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=256)
    tokenized_inputs['labels'] = batch['label']
    return tokenized_inputs

train_dataset = dataset['train'].map(tokenize, batched=True)
val_dataset = dataset['validation'].map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 加载序列分类模型，并指定类别数
model = AutoModelForSequenceClassification.from_pretrained("gpt2", 
                                                           num_labels=3,  # 假设有3个类别
                                                           cache_dir=download_and_save_dir)

model.config.pad_token_id = model.config.eos_token_id
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻最后三层 Transformer 层
for layer in model.transformer.h[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

# 解冻分类器层的权重和偏置
model.score.weight.requires_grad = True  # 使用 score 而不是 lm_head

# 使用适当的数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/ckpt',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 保存微调后的模型和 tokenizer
model.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model')
tokenizer.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model')

print("saved")