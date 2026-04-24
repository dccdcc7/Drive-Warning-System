import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


# 0. 加载数据集
from datasets import Dataset
import json
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    def format_example(example):
        if example.get("input", "") == "":
            prompt = f"Human: {example['instruction']}\n\nAssistant: "
        else:
            prompt = f"Human: {example['instruction']}\n\n{example['input']}\n\nAssistant: "
        return {"text": prompt + example["output"]}
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)
    return dataset
# 加载数据
train_dataset = load_dataset("train_data.json")
eval_dataset = load_dataset("eval_data.json")

# 1. 加载模型和分词器
# model_name = r"./model/Qwen/Qwen2.5-0.5B-Instruct"
# model_name = r"D:\python\pycharmproject\QW25\model\Qwen\Qwen2.5-0.5B-Instruct"
from pathlib import Path
model_path = Path(r"D:\python\pycharmproject\QW2.5-0.5B\models\Qwen\Qwen2___5-0___5B-Instruct").resolve()
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# 2. 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,              # 秩，建议32-128
    lora_alpha=16,     # 缩放参数
    lora_dropout=0.1,  # dropout
    target_modules=[   # 目标模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
)

# 3. 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数

# 4. 数据预处理
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_overflowing_tokens=False,
    )
    return tokenized

# 假设已经有训练数据
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./qwen2.5-0.5b-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to=["tensorboard"],
)

# 6. 数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 7. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. 开始训练
trainer.train()

# 9. 保存模型
model.save_pretrained("./qwen2.5-0.5b-lora-final")
tokenizer.save_pretrained("./qwen2.5-0.5b-lora-final")