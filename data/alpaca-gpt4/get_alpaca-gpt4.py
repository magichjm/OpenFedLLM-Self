import json
from datasets import load_dataset
import os

dataset_name_with_namespace = "vicgalle/alpaca-gpt4"
ds = load_dataset(dataset_name_with_namespace)

# 获取数据集的最后一个目录名作为数据集名
dataset_name = dataset_name_with_namespace.split('/')[-1]

# 遍历数据集的第一级划分
for split_name, split_data in ds.items():
    print(f"\n--- 划分: {split_name} ---")

    # 输出基本信息：数据量
    print(f"  数据量: {len(split_data)}")

    # 输出第一条数据
    if len(split_data) > 0:
        print("  第一条数据:")
        print(split_data[0])
    else:
        print("  该划分没有数据。")

    # 将当前划分的数据转换为 JSON 并保存
    output_filename = f"{dataset_name}_{split_name}.json"
    data_to_save = split_data.to_list()  # 将 Dataset 对象转换为 Python 列表

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"  数据已保存到文件: {output_filename}")
    except Exception as e:
        print(f"  保存数据到文件时发生错误: {e}")

print("\n操作完成！")