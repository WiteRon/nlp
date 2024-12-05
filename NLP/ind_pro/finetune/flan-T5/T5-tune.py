# coding:utf-8
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# 加载数据集
dataset = load_dataset('json', data_files={
    'train': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_train_5000.json',
    'validation': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_val_1000.json'
})

# 加载 flan-T5 tokenizer
download_and_save_dir = "/home/zshiap/NLP/ind_pro/data/llms/flan-T5"
model_name = "google/flan-t5-base"  # 或者选择其他大小的 flan-T5 模型，如 "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_and_save_dir)

# 数据预处理
def tokenize(batch):
    tokenized_inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=256)
    tokenized_inputs['labels'] = batch['label']
    return tokenized_inputs

train_dataset = dataset['train'].map(tokenize, batched=True)
val_dataset = dataset['validation'].map(tokenize, batched=True)

# 设置数据格式
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 加载序列分类模型，并指定类别数
model = T5ForSequenceClassification.from_pretrained(model_name,
                                                    num_labels=3,  # 假设有3个类别
                                                    cache_dir=download_and_save_dir)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻分类器头部
model.classification_head.dense.weight.requires_grad = True
model.classification_head.dense.bias.requires_grad = True
model.classification_head.out_proj.weight.requires_grad = True
model.classification_head.out_proj.bias.requires_grad = True

# 使用适当的数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='/home/zshiap/NLP/ind_pro/finetune/models/T5/ckpt',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,  # 控制台输出的日志步长，例如每10步输出一次
    learning_rate=3e-4,  # 根据官方文档调整学习率
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

# 定义一个简单的回调类来打印进度和loss
from transformers.trainer_callback import TrainerCallback

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Epoch: {state.epoch:.2f}, Step: {state.global_step}, Loss: {logs['loss']}")

# 添加回调
trainer.add_callback(PrintLossCallback())

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 保存微调后的模型和 tokenizer
model.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/T5/model')
tokenizer.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/T5/model')



print("saved")