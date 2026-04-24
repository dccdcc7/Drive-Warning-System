# # llm_advice_service.py
# import threading, queue, time, json
#
#
# # ---------- 1. 线程安全队列 ----------
# prompt_q = queue.Queue(maxsize=2)   # 主 -> LLM
# advice_q = queue.Queue(maxsize=2)   # LLM -> 主
#
# # ---------- 2. LLM 线程：永久循环 ----------
# def llm_worker(model_path="./Qwen2.5-0.5B-Instruct"):
#         # device = "cuda" if torch.cuda.is_available() else "cpu"
#         # tok = AutoTokenizer.from_pretrained(model_path)
#         # model = AutoModelForCausalLM.from_pretrained(model_path,
#         #                                              torch_dtype=torch.float16,
#         #                                              device_map=device)
#         #
#         # while True:
#         #     item = prompt_q.get()      # 阻塞等 prompt
#         #     if item is None:           # 毒丸退出
#         #         break
#         #
#         #     prompt = item
#         #     inputs = tok(prompt, return_tensors="pt").to(device)
#         #     with torch.no_grad():
#         #         out = model.generate(**inputs,
#         #                              max_new_tokens=40,
#         #                              do_sample=False,
#         #                              eos_token_id=tok.eos_token_id,
#         #                              pad_token_id=tok.eos_token_id)
#         #     advice = tok.decode(out[0][inputs.input_ids.shape[1]:],
#         #                         skip_special_tokens=True)
#         #     advice_q.put(advice)       # 回传主线程
#         advice = "fuck u"
#         time.sleep(0.5)
#         advice_q.put(advice)  # 回传主线程
#
# # ---------- 3. 启动函数 ----------
# def start():
#     threading.Thread(target=llm_worker, daemon=True).start()
#
# # ---------- 4. 主线程调用 ----------
# def send_scene(speed, weather):
#     """非阻塞发送当前场景"""
#     prompt = speed
#     try:
#         prompt_q.put_nowait(prompt)
#     except queue.Full:
#         pass          # 丢帧保实时
#
# def get_advice_nonblock():
#     """非阻塞取回 LLM 结果"""
#     try:
#         return advice_q.get_nowait()
#     except queue.Empty:
#         return "None"

# llm_advice_service.py
# import threading, queue, time
#
# # ---------- 线程安全队列 ----------
# prompt_q = queue.Queue(maxsize=2)   # 主 -> 副线程（发请求用，可不用）
# advice_q = queue.Queue(maxsize=2)   # 副线程 -> 主（收结果用）
#
# # ---------- 副线程：每 0.5 s 硬塞一条 ----------
# def dummy_worker(i):
#     while True:
#         time.sleep(3)
#         advice_q.put("fuck u")      # 直接塞，不理会 prompt_q
#
# # ---------- 启动函数 ----------
# def start():
#     threading.Thread(target=dummy_worker, daemon=True).start()
#
# # ---------- 主线程调用 ----------
# def send_scene(speed, weather):
#     """非阻塞发送（本示例已无用，可留空）"""
#     pass
#
# def get_advice_nonblock():
#     """非阻塞取结果"""
#     try:
#         return advice_q.get_nowait()
#     except queue.Empty:
#         return "None"

# llm_advice_service.py
import threading, queue, time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 线程安全队列 + 最近一次值缓存
cnt_q = queue.Queue(maxsize=2)
_last_cnt = "AI驾驶伴侣启动中..."          # 缓存：主线程永远读它
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("模型会运行在：", device)

# 加载模型和分词器
model_path = "./QW25/models/Qwen/Qwen2___5-0___5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 副线程：收到就更新缓存
def cache_worker():
    global _last_cnt
    while True:
        try:
            # 阻塞 0.1 s 超时，防止死锁
            prompt = cnt_q.get(timeout=0.1)
            #模拟QW推理的时间
            #QW
            #prompt = input("请输入您的问题或指令：").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            messages = [
                {'role': 'system', 'content': '你是驾驶预警助手，提供驾驶相关的建议和预警，简洁'},
                {'role': 'user', 'content': prompt}
            ]
            # 构建对话历史
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors='pt').to(device)

            # 生成回复
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs['attention_mask'],  # ← 关键行
                max_new_tokens=35,
                # 可选参数
                # do_sample=False,
                # temperature=0.7,
                # top_k=50,
                # top_p=0.95,
                # repetition_penalty=1.2,
            )

            # 提取生成的部分
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                             zip(model_inputs.input_ids, generated_ids)]
            # 解码为文本
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #这里的代码后面就换成QW推理的代码了，最终返回的就是字符串
            _last_cnt = response
        except queue.Empty:
            continue

# 启动
def start():
    threading.Thread(target=cache_worker, daemon=True).start()

# 主线程调用
def send_frame_cnt(cnt: int):
    """非阻塞发送当前帧号"""
    try:
        cnt_q.put_nowait(cnt)
    except queue.Full:
        pass

def get_frame_cnt_cached() -> int:
    """永远返回最近一次收到的 frame_cnt（队列空也不丢值）"""
    return _last_cnt