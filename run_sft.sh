max_steps=30
num_rounds=200
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=5
sample_clients=2
lora_r=8
lora_alpha=16   # twice of lora_r
lr=5e-5

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="vicgalle/alpaca-gpt4"
dataset_sample=50000
model_name_or_path="./base/Llama-2-7b-hf"
output_dir=./output

gpu=0
fed_alg="fedavg"
template_name="alpaca"  # 先定义 template 变量

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template $template_name