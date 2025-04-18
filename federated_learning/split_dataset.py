import random


def split_dataset(fed_args, script_args, dataset):
    """
    将原始数据集按照联邦学习客户端进行划分（当前支持 IID 分布）。
    返回长度为 num_clients 的列表，每个元素是对应客户端的数据子集
    """
    # 先打乱整个数据集，确保划分时是随机均匀的
    dataset = dataset.shuffle(seed=script_args.seed)

    # 初始化客户端本地数据列表
    local_datasets = []

    # 当前仅支持 IID 分布（平均划分）
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            # shard: 把数据平均分成 num_clients 份，当前客户端取第 i 份
            local_datasets.append(dataset.shard(num_shards=fed_args.num_clients, index=i))

    return local_datasets


def get_dataset_this_round(dataset, round, fed_args, script_args):
    """
    从整个客户端本地数据集中采样一小部分子集，用于当前轮训练
    """
    # 计算当前轮所需样本总数 = 每轮训练总步数 × 每步的总样本数
    num2sample = (
            script_args.batch_size
            * script_args.gradient_accumulation_steps
            * script_args.max_steps
    )

    # 避免采样数量超过数据集总量
    num2sample = min(num2sample, len(dataset))

    # 以 round 作为随机种子，确保每轮采样是确定性的（可复现），但又不重复
    random.seed(round)

    # 从数据集中随机选出 num2sample 个索引
    random_idx = random.sample(range(0, len(dataset)), num2sample)

    # 根据随机索引从数据集中选取样本，构成当前轮的数据子集
    dataset_this_round = dataset.select(random_idx)

    # 返回这一轮要用的数据集子集
    return dataset_this_round