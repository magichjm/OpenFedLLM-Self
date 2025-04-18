import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def merge_lora(base_model_name, lora_path):
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # 将 LoRA 微调应用到基础模型上
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    # 加载对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # 合并 LoRA 权重到基础模型中
    model = peft_model.merge_and_unload()

    # 获取 checkpoint 的上一级目录
    parent_dir = os.path.dirname(lora_path)

    # 构造新的保存路径，作为 checkpoint 同级的 full 目录
    target_model_path = os.path.join(parent_dir, "full")
    os.makedirs(target_model_path, exist_ok=True)

    # 保存模型和分词器
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)

    print(f"合并后的模型已保存至: {target_model_path}")


if __name__ == "__main__":
    # 在这里直接设置变量
    base_model_path = "../base/Llama-2-7b-hf"  # 基础模型
    lora_path = "../output/alpaca-gpt4_50000_fedavg_c5s2_i30_b16a1_l512_r8a16_20250416000539/checkpoint-200"  # LoRA 权重路径
    # 会把合并后的模型保存到 checkpoint 的同级 full 目录
    merge_lora(base_model_path, lora_path)
