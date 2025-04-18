import os
import json
from collections import defaultdict
from statistics import mean

# 假设你的 JSON 文件都在这个目录下
folder_path = 'model_judgment'

# 用来统计每个 judger 的分数列表
judger_scores = defaultdict(list)

# 遍历文件夹中的所有 JSON 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                judger = item.get('judger')
                rating = item.get('rating')
                try:
                    score = float(rating)
                    judger_scores[judger].append(score)
                except ValueError:
                    continue  # 忽略无法转换为数字的评分

# 输出每个 judger 的评分分布与平均分
for judger, scores in judger_scores.items():
    score_distribution = defaultdict(int)
    for score in scores:
        score_distribution[score] += 1
    average_score = mean(scores)

    print(f"Judger: {judger}")
    print(f"评分分布: {dict(score_distribution)}")
    print(f"平均分: {average_score:.2f}")
    print("-" * 40)
