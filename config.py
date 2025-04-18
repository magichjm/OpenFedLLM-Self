from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=100, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=5, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    # 每一轮全局通信都保存全局global
    save_model_freq: Optional[int] = field(default=1, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})

@dataclass
class ScriptArguments:
    # 默认llm模型
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    # 默认数据集name，传入这个会使用hf的datasets库去读取，而不需要本地数据集路径了
    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca-gpt4 use 2e-5
    # 客户端本地训练的bs
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    # 客户端本地训练的梯度累计步数
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    # 默认使用peft
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    # 模型保存文件的顶级输出路径
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    # 客户端本地训练时的trainer输出频率，默认1w表示不输出
    logging_steps: Optional[int] = field(default=10000, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    # 客户端本地训练轮数
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    # 客户端本地训练步数
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    # 客户端训练时保存模型所需要的步数，默认1w表示不保存
    save_steps: Optional[int] = field(
        default=10000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    # 限制客户端每次训练保存的checkpint，没啥用，全局保存的不受他影响
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    # 启用梯度检查点能够减少梯度显存占用，但是会降低训练速度
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    # 组织训练数据(instruction、response)的模板
    template: Optional[str] = field(default="alpaca-gpt4", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    # 从原始数据集中采样全局数据集的大小，所有客户端数据之和
    dataset_sample: Optional[int] = field(default=50000, metadata={"help": "the number of samples to use from the dataset"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})

parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # 不设置target_modules，使用默认的配置，只有q、v
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = torch.bfloat16
    return device_map, quantization_config, torch_dtype

# 生成新的outputdir并生成args.json
def save_config(script_args, fed_args):
    # 获取当前时间，并格式化为字符串，作为输出路径的一部分，用于区分不同实验
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    # 获取数据集文件名（去除路径），用于生成唯一的输出目录
    dataset_name_split = os.path.basename(script_args.dataset_name)

    # 构建输出目录的路径，包含了数据集名、采样方式、联邦算法、客户端数目、采样客户端数、
    # 最大训练步数、batch大小、梯度累积、序列长度、LoRA参数等，方便后续查阅
    output_dir = f"{script_args.output_dir}/{dataset_name_split}_{script_args.dataset_sample}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    # 保证目录唯一性，防止重名覆盖：如果已存在，则更新时间戳，直到生成一个不存在的目录
    while True:
        if not os.path.exists(output_dir):
            # 如果该输出目录不存在，则创建它
            os.mkdir(output_dir)
            break
        else:
            # 如果已存在（极少情况），则将时间加1秒，重新生成路径
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{now_time}"

    # 将生成的唯一输出目录回写给 script_args，以供后续使用
    script_args.output_dir = output_dir
    # 将 script_args 和 fed_args 中的参数转为字典，保存为 JSON 格式文件到 output_dir/args.json
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),  # 转为 dict 方便 JSON 序列化
            "fed_args": asdict(fed_args)
        }
        json.dump(combined_dict, f, indent=4)  # 缩进格式化写入文件