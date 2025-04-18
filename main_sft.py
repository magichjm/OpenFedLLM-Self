import copy
import os
import logging
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

from pprint import pprint
from dataclasses import asdict

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
# 重定义output_dir参数，并保存args.json
os.makedirs(script_args.output_dir, exist_ok=True)
save_config(script_args, fed_args)
print(f">> ===== Output Directory =====\n{script_args.output_dir}")

# ===== Configure logging =====
log_file_path = os.path.join(script_args.output_dir, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_file_path)  # Output to log file
    ]
)

# 输出args
logging.info(">> ===== Script Args =====")
for k, v in asdict(script_args).items():
    logging.info(f"{k:30s}: {v}")

logging.info(">> ===== Fed Args =====")
for k, v in asdict(fed_args).items():
    logging.info(f"{k:30s}: {v}")


# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
# 划分数据集
local_datasets = split_dataset(fed_args, script_args, dataset)
# 统计每个客户端的数据量
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
# 打印每个客户端的数据量
logging.info(">> ===== Sample count per client =====")
for i, num_samples in enumerate(sample_num_list):
    logging.info(f"Client {i:2d}: {num_samples} samples")


# ===== Get model config =====
# 根据args的量化配置得到模型配置
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# 加载llm模型
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

# 如果使用量化
if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
# 配置lora层
model = get_peft_model(model, peft_config)

# 输出模型结构和可训练参数
logging.info(model)
model.print_trainable_parameters()

# 训练阶段禁用kv缓存，推理阶段再开
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# 如果开了梯度检查点就需要打开这个，但是现在还不懂？
if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

# ===== Define the global and local models =====4
# 读取全局lora参数
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
# 为每个client设置lora参数
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token    # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====

# 根据template获取编码函数（后续会传入trainer中，对数据集进行编码）
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# 把“Response template”转成 token ids
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
# DataCollatorForCompletionOnlyLM能自动 mask 掉 prompt 部分和response_template_ids，只对真正的response计算loss
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
# client*round的loss列表
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)
    logging.info(f">> ==================== Round {round+1} : Clients {clients_this_round} ====================")
    round_losses = {}
    for client in range(fed_args.num_clients):
        # 不参与训练就是-1代替
        if client not in clients_this_round:
            training_loss[client].append(-1)         # -1 is an indicator of not training
            continue

        logging.info(f">> Round {round+1} - Client {client}: Training started.")

        # 客户端获取服务器全局参数
        set_peft_model_state_dict(model, global_dict)    # sync the global model to the local model
        # 客户端选择本轮参与训练的数据
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)     # get the required sub-dataset for this round
        # 输出当前轮次客户端使用的数据大小
        logging.info(f">> Round {round+1} - Client {client}: Using {len(sub_dataset)} samples for training.")
        # 学习率调度器
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        # 获取训练参数
        training_args = get_training_args(script_args, new_lr)
        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        # results.training_loss是训练过程中每一个step的平均loss
        training_loss[client].append(results.training_loss)
        round_losses[client] = results.training_loss

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))    # copy is needed!

        logging.info( f">> Round {round + 1} - Client {client}: Training finished with loss: {results.training_loss:.4f}\n")

    # Output avg round losses
    avg_loss = sum(round_losses.values()) / len(round_losses) if round_losses else 0
    logging.info(f">> Round {round+1} - Average loss of participating clients: {avg_loss:.4f}")

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list,
        clients_this_round, round, proxy_dict=proxy_dict,
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    # 更新全局参数
    set_peft_model_state_dict(model, global_dict)    # Update global model

    # ===== Save the model =====
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

# Save training loss to TXT
output_file_path = os.path.join(script_args.output_dir, "training_loss.txt")
with open(output_file_path, 'w') as f:
    for client_id, losses in enumerate(training_loss):
        loss_str = '[' + ','.join(map(str, losses)) + ']'
        line = f'client{client_id}:{loss_str}\n'
        f.write(line)
logging.info(f">> Training loss saved to: {output_file_path},FL-LLM finished!")