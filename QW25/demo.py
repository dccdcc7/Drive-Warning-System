import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("模型会运行在：", device)

# 加载模型和分词器
model_path = "./models/Qwen/Qwen2___5-0___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 用户输入
# prompt = "写一个故事"


while True:
    prompt = input("请输入您的问题或指令：").strip()
    if prompt.lower() in {"exit", "quit"}:
        break
    messages = [
        {'role': 'system', 'content': '你是安全驾驶助手，给出驾驶命令'},
        {'role': 'user', 'content': prompt}
    ]
    # 构建对话历史
    time1 = time.time()
    # 格式化对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(device)

    # 分词并转移到设备


    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs['attention_mask'],   # ← 关键行
        max_new_tokens=35,
        # 可选参数
        # do_sample=False,
        # temperature=0.7,
        # top_k=50,
        # top_p=0.95,
        # repetition_penalty=1.2,
    )

    # 提取生成的部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    # 解码为文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(type(response))
    time2 = time.time()
    print("推理时间：",time2-time1)
    print(response)