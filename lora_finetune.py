import os 
import json
import torch
import argparse
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

parser = argparse.ArgumentParser(description="classfiy")
parser.add_argument("--cache_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--csv_file_path", type=str, required=True)
args = parser.parse_args()


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"User: {example['instruction']+example['input']}\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"Assistant: {example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 导入数据
df = pd.read_csv(args.csv_file_path, encoding='utf-8')
all_data = df.to_dict(orient='records')
instruction_data = []

TASK = """
你是一个序列文本分类助手，一共有七个类别，分别为：仇恨辱骂、伦理道德、偏见歧视、身心健康、违法犯罪、隐私财产、谣言错误；
你将收到一条文本信息，请对当前文本进行分类，要求仅输出类别信息：
"""

for item in all_data:
    id, seed, scene = item['id'], item['seed'], item['scene']
    instruction_data.append({
        'instruction': TASK, 
        'input': seed, 
        'output': scene
    })

df = pd.DataFrame(instruction_data)
ds = Dataset.from_pandas(df)

# 创建模型
model_dir = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_dir)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.enable_input_require_grads()              # 开启梯度检查点时，要执行该方法
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = 'right'
print('=> model and tokenizer is ok...')

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(tokenized_id[0]['input_ids'])
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

# 定义LORA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
print('=> lora config is ok...')
model = get_peft_model(model, config)
logs = model.print_trainable_parameters()
print(logs)

# 配置训练参数
args = TrainingArguments(
    output_dir=args.cache_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()