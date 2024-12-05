# coding:utf-8
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# �������ݼ�
dataset = load_dataset('json', data_files={
    'train': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_train_5000.json',
    'validation': '/home/zshiap/NLP/ind_pro/data/multi_nli_json/new_val_1000.json'
})

# ���� flan-T5 tokenizer
download_and_save_dir = "/home/zshiap/NLP/ind_pro/data/llms/flan-T5"
model_name = "google/flan-t5-base"  # ����ѡ��������С�� flan-T5 ģ�ͣ��� "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_and_save_dir)

# ����Ԥ����
def tokenize(batch):
    tokenized_inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=256)
    tokenized_inputs['labels'] = batch['label']
    return tokenized_inputs

train_dataset = dataset['train'].map(tokenize, batched=True)
val_dataset = dataset['validation'].map(tokenize, batched=True)

# �������ݸ�ʽ
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# �������з���ģ�ͣ���ָ�������
model = T5ForSequenceClassification.from_pretrained(model_name,
                                                    num_labels=3,  # ������3�����
                                                    cache_dir=download_and_save_dir)

# �������в���
for param in model.parameters():
    param.requires_grad = False

# �ⶳ������ͷ��
model.classification_head.dense.weight.requires_grad = True
model.classification_head.dense.bias.requires_grad = True
model.classification_head.out_proj.weight.requires_grad = True
model.classification_head.out_proj.bias.requires_grad = True

# ʹ���ʵ�������������
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ����ѵ������
training_args = TrainingArguments(
    output_dir='/home/zshiap/NLP/ind_pro/finetune/models/T5/ckpt',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,  # ����̨�������־����������ÿ10�����һ��
    learning_rate=3e-4,  # ���ݹٷ��ĵ�����ѧϰ��
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

# ����һ���򵥵Ļص�������ӡ���Ⱥ�loss
from transformers.trainer_callback import TrainerCallback

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Epoch: {state.epoch:.2f}, Step: {state.global_step}, Loss: {logs['loss']}")

# ��ӻص�
trainer.add_callback(PrintLossCallback())

# ��ʼѵ��
trainer.train()

# ����ģ��
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# ����΢�����ģ�ͺ� tokenizer
model.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/T5/model')
tokenizer.save_pretrained('/home/zshiap/NLP/ind_pro/finetune/models/T5/model')



print("saved")