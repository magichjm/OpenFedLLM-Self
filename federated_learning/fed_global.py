import random
import torch


# 获取当前轮参与训练的客户端
def get_clients_this_round(fed_args, round):
    # 如果算法是 local 类型，只选择固定的某一个客户端（如 local1 表示客户端编号为1）
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        # 如果总客户端数比采样数还少，就全选
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            # 否则，从所有客户端中随机采样 sample_clients 个
            random.seed(round)  # 使用当前轮数作为随机种子，保证可复现
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round  # 返回本轮选中的客户端编号列表


# 全局模型聚合函数，根据不同的联邦算法进行聚合
def global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list,
        clients_this_round, round_idx, proxy_dict=None,
        opt_proxy_dict=None, auxiliary_info=None):
    # 计算当前轮参与客户端总样本数（用于加权平均）
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None  # 初始化辅助变量（只在 Scaffold 用到）

    # Scaffold 算法：使用控制变量辅助训练
    if fed_args.fed_alg == 'scaffold':
        for key in global_dict.keys():
            # 按样本数加权平均聚合模型参数
            global_dict[key] = sum([
                local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])

        # 聚合控制变量（control variate）
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        for key in global_auxiliary.keys():
            delta_auxiliary = sum([
                auxiliary_delta_dict[client][key] for client in clients_this_round
            ])
            # 控制变量按客户端数量归一化
            global_auxiliary[key] += delta_auxiliary / fed_args.num_clients

    # FedAvgM：引入动量的联邦平均
    elif fed_args.fed_alg == 'fedavgm':
        for key in global_dict.keys():
            # 计算每个参数的平均变化量（delta）
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])
            # 使用动量更新公式
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] +
                (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            # 更新全局模型
            global_dict[key] = global_dict[key] + proxy_dict[key]

    # FedAdagrad：自适应学习率的优化方法
    elif fed_args.fed_alg == 'fedadagrad':
        for key, param in opt_proxy_dict.items():
            # 平均每个客户端的参数变化
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key])
                for client in clients_this_round
            ]) / len(clients_this_round)
            # 记录更新量
            proxy_dict[key] = delta_w
            # 更新累积平方梯度
            opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
            # 用自适应学习率更新模型
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    # FedYogi：在 Adagrad 基础上加入了对梯度平方的控制（抑制爆炸）
    elif fed_args.fed_alg == 'fedyogi':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key])
                for client in clients_this_round
            ]) / len(clients_this_round)
            # 更新一阶动量
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] +
                (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            # 控制平方项的更新方式（Yogi 方法核心）
            delta_square = torch.square(proxy_dict[key])
            opt_proxy_dict[key] = param - (
                    (1 - fed_args.fedopt_beta2) * delta_square * torch.sign(param - delta_square)
            )
            # 更新全局模型
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    # FedAdam：类似于 Adam 优化器，用一阶、二阶矩估计
    elif fed_args.fed_alg == 'fedadam':
        for key, param in opt_proxy_dict.items():
            delta_w = sum([
                (local_dict_list[client][key] - global_dict[key])
                for client in clients_this_round
            ]) / len(clients_this_round)
            # 一阶动量
            proxy_dict[key] = (
                fed_args.fedopt_beta1 * proxy_dict[key] +
                (1 - fed_args.fedopt_beta1) * delta_w
                if round_idx > 0 else delta_w
            )
            # 二阶动量（平方梯度的指数加权平均）
            opt_proxy_dict[key] = (
                    fed_args.fedopt_beta2 * param +
                    (1 - fed_args.fedopt_beta2) * torch.square(proxy_dict[key])
            )
            # 更新模型参数
            global_dict[key] += fed_args.fedopt_eta * torch.div(
                proxy_dict[key], torch.sqrt(opt_proxy_dict[key]) + fed_args.fedopt_tau
            )

    # 默认使用 FedAvg：按样本数加权平均聚合
    else:
        for key in global_dict.keys():
            global_dict[key] = sum([
                local_dict_list[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ])

    return global_dict, global_auxiliary  # 返回更新后的模型参数和辅助变量（仅 Scaffold 用）
