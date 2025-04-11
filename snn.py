import torch
import torch.nn as nn
import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
from utils.model_utils import create_model_instance # 假设这个函数能用
import copy

@torch.no_grad()
def add_model_inplace(param_to_add, target_param):
    if param_to_add.shape == target_param.shape:
        target_param += param_to_add
    else:
        try:
            if target_param.dim() == 4: # 卷积层 [out, in, H, W]
                target_param[:param_to_add.shape[0], :param_to_add.shape[1], :, :] += param_to_add
            elif target_param.dim() == 2: # 全连接层 [out, in]
                target_param[:param_to_add.shape[0], :param_to_add.shape[1]] += param_to_add
            elif target_param.dim() == 1: # 偏置项 [out]
                target_param[:param_to_add.shape[0]] += param_to_add
            else:
                pass 
        except IndexError as e:
            print(f"Error in add_model_inplace: {e}. Shapes: param_to_add={param_to_add.shape}, target_param={target_param.shape}")
            raise e

@torch.no_grad()
def scale_model(param, scale):
    return param * scale

@torch.no_grad()
def mask_model_inplace(target_param, width_fraction):
    """原地将 target_param 的左上角 width_fraction 部分置零"""
    if width_fraction == 0.0:
        return
    try:
        if target_param.dim() == 4: # [out, in, H, W]
            out_slice = int(target_param.shape[0] * width_fraction)
            in_slice = int(target_param.shape[1] * width_fraction)
            target_param[:out_slice, :in_slice, :, :] = 0
        elif target_param.dim() == 2: 
            out_slice = int(target_param.shape[0] * width_fraction)
            in_slice = int(target_param.shape[1] * width_fraction)
            target_param[:out_slice, :in_slice] = 0
        elif target_param.dim() == 1: 
            out_slice = int(target_param.shape[0] * width_fraction)
            target_param[:out_slice] = 0
        else:
            pass 
    except IndexError as e:
         print(f"Error in mask_model_inplace: {e}. Shape: {target_param.shape}, width_fraction: {width_fraction}")
         raise e


@torch.no_grad()
def get_ring(param, lower_bound, upper_bound):
    mask_model_inplace(param, lower_bound)
    copy_model = copy.deepcopy(param)
    mask_model_inplace(copy_model, upper_bound) # 留下外层环
    add_model_inplace(scale_model(copy_model, -1), param) # param 掩盖掉外层环


@torch.no_grad()
def model_aggregation(clients, client_widths_input, global_model):
    """
    聚合不同宽度的客户端模型。

    Args:
        clients: 客户端对象列表，每个对象应有 .model, .width, .datasize 属性。
        client_widths_input: 客户端宽度列表 (仅用于确定宽度边界)。
    """
    # 设置宽度边界和 ring 累加器，多少个宽度就有多少个 ring
    widths_set = set(client_widths_input)
    ring_accumulators = [
        create_model_instance("alexnet", "cifar10", width=1.0)
        for _ in widths_set # Create one accumulator per lower bound width
    ]
    widths_set.add(0.0)
    widths_set.add(1.0)
    sorted_widths = sorted(list(widths_set)) # e.g., [0.0, 0.25, 0.5, 1.0]
    # 计算每个环的加权和，并进行掩码
    for i, cur_min_width in enumerate(sorted_widths):
        available_clients = [c for c in clients if c.width > cur_min_width]
        if not available_clients:
            # 如果没有客户端满足条件 (例如 cur_min_width=1.0 时)，
            # 设置为 global_model 对应的值
            if cur_min_width == 1.0:
                continue
            accumulator_model = ring_accumulators[i]
            for param in accumulator_model.parameters():
                get_ring(param, cur_min_width, sorted_widths[i+1])
            continue # 跳到下一个宽度边界

        # 计算权重
        total_datasize = sum(client.datasize for client in available_clients)
        client_weights = {client: client.datasize / total_datasize for client in available_clients}
        print(f"当前环 ({cur_min_width}, {sorted_widths[i+1]}) 的客户端权重: {client_weights}")
        # 获取当前环的累加器模型
        accumulator_model = ring_accumulators[i]

        # 遍历累加器模型的每个参数
        for name, accum_param in accumulator_model.state_dict().items():
            accum_param.zero_()
            for client in available_clients:
                if name in client.model.state_dict(): # 确保客户端有这个参数
                    client_param = client.model.state_dict()[name]
                    scaled_client_param = scale_model(client_param, client_weights[client])
                    # 将缩放后的客户端参数加到累加器的对应子块
                    add_model_inplace(scaled_client_param, accum_param)
            # mask_model_inplace(accum_param, cur_min_width)
            get_ring(accum_param, cur_min_width, sorted_widths[i+1]) # 掩码当前环

    final_model = create_model_instance("alexnet", "cifar10", width=1.0)
    for param in final_model.parameters():
        param.zero_()
    for i in range(len(sorted_widths) - 1): # 最后一个宽度边界不需要计算环
        ring_model = ring_accumulators[i]
        for name, final_param in final_model.named_parameters():
            if name in ring_model.state_dict():
                ring_param = ring_model.state_dict()[name]
                add_model_inplace(ring_param, final_param)

    return final_model

# --- 示例用法 ---
class MockClient:
    def __init__(self, width, datasize):
        self.width = width
        self.datasize = datasize
        self.model = create_model_instance("alexnet", "cifar10", width=self.width)
        # 假设模型已训练或有参数
        for param in self.model.parameters():
            # param.data.uniform_(-0.1, 0.1) # 随机填充
            # 全部设置为 1.0
            param.data.fill_(0.5) # 仅用于测试

if __name__ == '__main__':
    clients_list = [
        MockClient(width=0.25, datasize=100),
        MockClient(width=0.5, datasize=200),
        MockClient(width=0.5, datasize=150),
        MockClient(width=1.0, datasize=300),
    ]
    client_widths = [c.width for c in clients_list]

    print("开始聚合 (修正后)...")
    global_model = create_model_instance("alexnet", "cifar10", width=1.0)
    print(client_widths)
    aggregated_final_model = model_aggregation(
        clients_list, client_widths, global_model)
    print("聚合完成.")

    all_zero = True
    for name, param in aggregated_final_model.named_parameters():
        if torch.any(param != 0):
            all_zero = False
            print(f"参数 {name} 非零 (norm: {torch.norm(param, p=torch.inf)})")
            # break # 可以只检查第一个非零
    if all_zero:
        print("警告：聚合后的模型所有参数仍然为零！")