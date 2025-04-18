# Step 1: 从 ModelScope 下载模型
from modelscope.hub.snapshot_download import snapshot_download

# 下载 shakechen/Llama-2-7b-hf 模型并保存到 ./llama2-7b 目录
model_dir = snapshot_download('shakechen/Llama-2-7b-hf',
                              cache_dir='./llama2-7b',
                              revision='master')

# Step 2: 加载模型和 tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.bfloat16).to(device)
print(model)

# Step 3: 打印显存使用情况
if torch.cuda.is_available():
    allocated_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mem = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"显存使用: allocated={allocated_mem:.2f} MB, reserved={reserved_mem:.2f} MB")
else:
    print("未使用GPU，正在使用CPU。")

# Step 4: 测试模型生成
input_text = "请简要介绍一下人工智能。"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=1000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("生成结果：")
print(generated_text)
