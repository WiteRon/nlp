# coding:utf-8

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# �������ݼ�
dataset = load_dataset('json', data_files={
    'train': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/train.json',
    'validation': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/validation.json'
})

download_and_save_dir = "/home/zshiap/NLP/ind_pro/data/llms/gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=download_and_save_dir)
# ���������Ϊ�������
tokenizer.pad_token = tokenizer.eos_token

# ����Ԥ����
def tokenize(batch):
    tokenized_inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=256)
    tokenized_inputs['labels'] = batch['label']
    return tokenized_inputs

train_dataset = dataset['train'].map(tokenize, batched=True)
val_dataset = dataset['validation'].map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# �������з���ģ�ͣ���ָ�������
model = AutoModelForSequenceClassification.from_pretrained("gpt2", 
                                                           num_labels=3,  # ������3�����
                                                           cache_dir=download_and_save_dir)

model.config.pad_token_id = model.config.eos_token_id
# �������в���
for param in model.parameters():
    param.requires_grad = False

# �ⶳ������� Transformer ��
for layer in model.transformer.h[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

# �ⶳ���������Ȩ�غ�ƫ��
model.score.weight.requires_grad = True  # ʹ�� score ������ lm_head

# ʹ���ʵ�������������
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ����ѵ������
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

# ����΢�����ģ�ͺ� tokenizer
model.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model')
tokenizer.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/100000_last_2/model')

print("saved")