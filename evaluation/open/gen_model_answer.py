import datasets
import json
from tqdm import tqdm
import os
import torch
import sys
import time

# 添加上级目录到模块路径
sys.path.append("../../")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils.template import TEMPLATE_DICT

# ========== 手动指定参数（代替 argparse） ==========
base_model_path = "../../output/alpaca-gpt4_50000_fedavg_c5s2_i30_b16a1_l512_r8a16_20250416000539/full"
lora_path = None
template_name = "alpaca"
use_vllm = True
bench_name = "vicuna"

# ========== 检查是否冲突 ==========
if use_vllm and lora_path is not None:
    raise ValueError("Cannot use both VLLM and LORA, need to merge the lora and then use VLLM")

# ========== 加载 Prompt 模板 ==========
template = TEMPLATE_DICT[template_name][0]
print(f">> You are using template: {template_name}\n{template}")

# ========== 加载评估数据集 ==========
if bench_name == "alpaca":
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    max_new_tokens = 2048
elif bench_name == "vicuna":
    eval_set = datasets.load_dataset("json", data_files="vicuna/question.jsonl")['train']

    def rename(example):
        example['instruction'] = example['turns'][0]
        return example

    eval_set = eval_set.map(rename)
    max_new_tokens = 2048
elif bench_name == "advbench":
    eval_set = datasets.load_dataset("csv", data_files="data/advbench/advbench.csv")["train"]
    eval_set = eval_set.rename_column("goal", "instruction")
    eval_set = eval_set.remove_columns(["target"])
    max_new_tokens = 1024
else:
    raise ValueError("Invalid benchmark name")

# ========== 构造模型名称 ==========
if lora_path:
    pre_str, checkpoint_str = os.path.split(lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(base_model_path)
    if last_str.startswith("full"):
        _, exp_name = os.path.split(pre_str)
        model_name = f"{exp_name}_full"
    else:
        model_name = last_str

# ========== 创建输出json文件 ==========
result_path = f"{bench_name}/model_answer/{model_name}.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
result_list = []

# ========== 使用 vLLM ==========
if use_vllm:
    # LLM类的初始化参数说明：
    # model: str (必填)  ···模型路径
    # tokenizer: Optional[str], default=None,如果未提供，通常会使用与模型相同的路径。
    # tokenizer_mode: str, default='auto'  ···分词器模式，示例: 'fast' 或 'slow'···
    # trust_remote_code: bool, default=False  ···是否信任远程代码，示例: True···
    # dtype: torch.dtype, default=torch.float16  ···加载模型的数据类型
    # max_seq_len: int, default=4096  ···最大输入序列长度
    # tensor_parallel_size: int, default=1  ···GPU张量并行数量
    # quantization: Optional[str], default=None  ···量化方式，示例: 'int8'···
    # seed: int, default=0  ···随机种子
    model = LLM(model=base_model_path)  # 初始化LLM对象

    if bench_name == "advbench":
        input_list = [template.format(example["instruction"] + '.', "", "")[:-1] for example in eval_set]
    else:
        input_list = [template.format(example["instruction"], "", "")[:-1] for example in eval_set]

    sampling_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=max_new_tokens)

    start_time = time.time()
    generations = model.generate(input_list, sampling_params)
    generations = [generation.outputs[0].text for generation in generations]

    # 输出生成的结果
    for i, example in tqdm(enumerate(eval_set)):
        example['output'] = generations[i]
        example['generator'] = model_name
        result_list.append(example)

    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / len(eval_set)

    # 输出最后的统计信息
    print("=" * 100)
    print(f">> Total Inference Time: {total_inference_time:.2f} seconds")
    print(f">> Average Inference Time per example: {avg_inference_time:.2f} seconds")
    print("=" * 100)

    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)

# ========== 使用普通推理 ==========
else:
    device = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
    ).to(device)

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    start_time = time.time()

    for i, example in enumerate(eval_set):
        if bench_name == "advbench":
            instruction = template.format(example["instruction"] + '.', "", "")[:-1]
        else:
            instruction = template.format(example["instruction"], "", "")[:-1]

        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)

        output_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=1.0,
            temperature=0.7
        )

        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)

        example['output'] = result
        example['generator'] = model_name

        print("=" * 100)
        print(f"[{i + 1}/{len(eval_set)}]")
        print(f"\nInput:\n{instruction}")
        print(f"\nOutput:\n{result}")
        print("=" * 100)

        result_list.append(example)
        with open(result_path, "w") as f:
            json.dump(result_list, f, indent=4)

    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time / len(eval_set)

    # 输出最后的统计信息
    print("=" * 100)
    print(f">> Total Inference Time: {total_inference_time:.2f} seconds")
    print(f">> Average Inference Time per example: {avg_inference_time:.2f} seconds")
    print("=" * 100)

print(f">> Done. Results saved to: {result_path}")
