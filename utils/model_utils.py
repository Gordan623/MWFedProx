import torch
import torch.nn as nn
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

import copy
from MWFedProx.models import MWAlexNet, LeNet

def create_model_instance(model_type, dataset_type, width = 1.0):
    """
    Create model instance by passing model type and dataset type.
    The model and dataset must collaborate, else will return None.
    Use init_model to initialize your own model structure.
    Or submitting pull request to add your own.
    """
    # print("My self-defined function")
    assert model_type and dataset_type is not None, f"Model type and dataset type cannot be None."
    models_dict = {      
        # ('mnist', 'alexnet'): AlexNet.AlexNetMnist,
        ('cifar10', 'alexnet'): MWAlexNet.MWAlexNet,
        # ('cifar10', 'alexnet2'): AlexNet.AlexNet2,    
        # ('image100', 'alexnet'): AlexNet.AlexNet_IMAGE,
        ('emnist', 'lenet'): LeNet.LeNet_Emnist, 
    }

    # 检查模型是否支持多宽度
    model = models_dict.get((dataset_type, model_type), None)
    if model:
        if hasattr(model, '__init__') and 'width' in model.__init__.__code__.co_varnames:
            return model(width = width)
        else:
            return model()
    else:
        return None

    # model = models_dict.get((dataset_type, model_type), None)
    # return model() if model else None

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
def add_model_inplace_reverse(param_to_add, target_param):
    """
    把较大的模型的param_to_add加到较小的模型的target_param上
    """
    if param_to_add.shape == target_param.shape:
        target_param += param_to_add
    else:
        try:
            if target_param.dim() == 4: # 卷积层 [out, in, H, W]
                # print shape
                # print(param_to_add.shape, target_param.shape)
                target_param += param_to_add[:target_param.shape[0], :target_param.shape[1], :, :]
            elif target_param.dim() == 2: # 全连接层 [out, in]
                target_param += param_to_add[:target_param.shape[0], :target_param.shape[1]]
            elif target_param.dim() == 1: # 偏置项 [out]
                target_param += param_to_add[:target_param.shape[0]]
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


@torch.no_grad()
def slice_model(model, width, device, model_type, dataset_type):
    """
    从大model中切片得到小model
    """
    reference = create_model_instance(model_type, dataset_type, width).to(device)
    for name, param in reference.named_parameters():
        param.data.zero_() # 初始化为0
        if name in model.state_dict():
            add_model_inplace_reverse(model.state_dict()[name], param)
    return reference



# def scale_model(model, scale):
#     """
#     scale the model parameters by a given scale factor.
#     Args:
#         model: The model to be scaled.
#         scale: The scale factor.
#     Returns:
#         model: The scaled model.
#     """
#     params = model.named_parameters()
#     dict_params = dict(params)
#     with torch.no_grad():
#         for name, param in dict_params.items():
#             dict_params[name].set_(dict_params[name].data * scale)
#     return model


# def model_divided(src_model, width, model_type, dataset_type):
#     """
#     根据源模型 src_model(宽度为1.0的全局模型)及目标宽度，
#     按照 FedSNN 的切分逻辑生成剪裁后的子模型。
#     适用于 FedProx 多宽度实现。
    
#     Args:
#         src_model: 全宽模型(width=1.0)——已加载全局参数
#         width: 目标宽度(<1 表示缩小)
#         model_type: 模型类型
#         dataset_type: 数据集类型
    
#     Returns:
#         dst_model: 剪裁后得到的子模型, 其参数已从src_model中切分获得
#     """
#     # 如果目标宽度为 1，不做剪裁，直接返回原模型
#     if width == 1:
#         return src_model
#     else:
#         # 根据模型类型构造目标宽度模型
#         dst_model = create_model_instance(model_type, dataset_type, width)
#         # dst_model = dst_model.to(src_model.device)  # 将目标模型移动到源模型所在的设备上
        
#         # 对目标模型做一次深拷贝(保证独立)
#         dst_model = copy.deepcopy(scale_model(dst_model, 0))
        
#         # 获取源模型和目标模型的参数
#         src_params = dict(src_model.named_parameters())
#         dst_params = dst_model.named_parameters()
        
#         with torch.no_grad():
#             for name, dst_param in dst_params:
#                 if name in src_params:
#                     src_param = src_params[name]
#                     # 针对不同的层作参数切分逻辑
#                     if src_param.dim() == 1:  # 情况一(1维张量且两者大小不等): 常见于卷积层的偏置项
#                         if dst_param.size(0) != src_param.size(0):  # 非输出层，取前一部分, 使用torch.split将较大的张量分割成与目标张量大小相等的部分, 然后将对应部分相加
#                             a1, a2 = src_param.split([dst_param.size(0), src_param.size(0)-dst_param.size(0)], dim=0)
#                             dst_param.copy_(a1.data)
#                             # dst_param.set_(a1.data + dst_param.data)
#                         else:  # 情况二(1维张量且大小相同): 输出层, 直接进行元素加法
#                             dst_param.copy_(src_param.data)
#                     elif src_param.dim() == 2:  # 情况三(2维张量，全连接层参数): 根据目标张量的尺寸在第1或第0维进行分割, 取出对应部分进行相加
#                         if dst_param.size(0) == src_param.size(0):  # 全连接层由1个维度分割(第0维)
#                             # 按输入维度分割
#                             a1, a2 = src_param.split([dst_param.size(1), src_param.size(1)-dst_param.size(1)], dim=1)
#                             dst_param.copy_(a1.data)
#                         else:  # 全连接层由两个维度分割(第0维&第1维)
#                             # 对于全连接层同时需要对输出维度和输入维度分割
#                             a1, a2 = src_param.split([dst_param.size(0), src_param.size(0)-dst_param.size(0)], dim=0)
#                             a3, a4 = a1.split([dst_param.size(1), src_param.size(1)-dst_param.size(1)], dim=1)
#                             dst_param.copy_(a3.data)
#                     else:  # 情况四(卷积层或池化层): 根据0维或者1维的分割来匹配输入参数的形状, 保证只有对应尺寸的部分被累加
#                         # 对于卷积层或其它多维层(形状：[out, in, ...])
#                         a1, a2 = src_param.split([dst_param.size(0), src_param.size(0)-dst_param.size(0)], dim=0)
#                         if src_param.size(1) > dst_param.size(1):
#                             a3, a4 = a1.split([dst_param.size(1), src_param.size(1)-dst_param.size(1)], dim=1)
#                             dst_param.copy_(a3.data)
#                         else:
#                             dst_param.copy_(a1.data)
#         return dst_model


# def add_model_with_different_width(dst_model, width1, src_model, width2):
#     """
#     将src_model的参数加到dst_model上
#     when width1 == width2: add straightly
#     when width1 != width2: split the bigger one then add their common part
#     Args:
#         dst_model: 目标模型
#         width1: 目标模型宽度
#         src_model: 源模型
#         width2: 源模型宽度
#     """
#     # 若宽度相同则直接逐层加和
#     if width1 == width2:
#         src_params = src_model.named_parameters()
#         dst_params = dst_model.named_parameters()
#         dst_dict = dict(dst_params)
#         with torch.no_grad():
#             for name, src_param in src_params:
#                 if name in dst_dict:
#                     dst_dict[name].set_(src_param.data + dst_dict[name].data)
#     else:
#         # 若宽度不同, 则需要分割各层参数, 只加共享部分
#         src_params = src_model.named_parameters()
#         dst_params = dst_model.named_parameters()
#         dst_dict = dict(dst_params)
#         with torch.no_grad():
#             for name, src_param in src_params:
#                 if name in dst_dict:
#                     dst_param = dst_dict[name]
#                     # 针对不同维度分别处理:
#                     if dst_param.dim() == 1:
#                         # 对于一维参数(偏置), 如果大小不同则取前target_size(src_param.size(0) - dst_param.size(0))部分加
#                         if dst_param.size(0) != src_param.size(0):
#                             a1, a2 = src_param.split([dst_param.size(0), src_param.size(0) - dst_param.size(0)], dim=0)
#                             dst_param.set_(a1.data + dst_param.data)
#                         else:
#                             dst_param.set_(src_param.data + dst_param.data)
#                     elif dst_param.dim() == 2:
#                         # 全连接层参数：如果行数(输出特征)一致, 则全连接层由1维分割, 对列数(第0维)切分
#                         if dst_param.size(0) == src_param.size(0):
#                             a1, a2 = src_param.split([dst_param.size(1), src_param.size(1) - dst_param.size(1)], dim=1)
#                             dst_param.set_(a1.data + dst_param.data)
#                         else:
#                             # 若输出尺寸不同，则先对第0维分割，再对第1维切分
#                             a1, a2 = src_param.split([dst_param.size(0), src_param.size(0) - dst_param.size(0)], dim=0)
#                             a3, a4 = a1.split([dst_param.size(1), src_param.size(1) - dst_param.size(1)], dim=1)
#                             dst_param.set_(a3.data + dst_param.data)
#                     else:
#                         # 对于卷积层或高维层
#                         a1, a2 = src_param.split([dst_param.size(0), src_param.size(0) - dst_param.size(0)], dim=0)
#                         if src_param.size(1) > dst_param.size(1):
#                             a3, a4 = a1.split([dst_param.size(1), src_param.size(1) - dst_param.size(1)], dim=1)
#                             dst_param.set_(a3.data + dst_param.data)
#                         else:
#                             dst_param.set_(a1.data + dst_param.data)
#     return dst_model


# def minus_model_with_differnt_width(dst_model, src_model):
#     """
#     计算 dst_model 和 src_model 之间的差值，即 dst_model - src_model
#     针对不同宽度的模型, 其对应层参数可能形状不同, 需要做切分处理
    
#     Args:
#       dst_model: 目标模型(较大宽度)
#       src_model: 源模型(较小宽度)
#     """
#     src_params = src_model.named_parameters()
#     dst_params = dst_model.named_parameters()
#     dst_dict = dict(dst_params)
#     with torch.no_grad():
#         for name, src_param in src_params:
#             if name in dst_dict:
#                 dst_param = dst_dict[name]
#                 if dst_param.dim() == 1:
#                     if dst_param.size(0) != src_param.size(0):  # 非输出层
#                         a1, a2 = dst_param.split([src_param.size(0), dst_param.size(0)-src_param.size(0)], dim=0)
#                         a1.set_(-src_param.data + a1.data)
#                         dst_param.set_(torch.cat([a1, a2], dim=0).data)
#                     else:  # 输出层
#                         dst_param.set_(-src_param.data + dst_param.data)
#                 elif dst_param.dim() == 2:
#                     if dst_param.size(0) == src_param.size(0):  # 全连接层由1个维度分割
#                         a1, a2 = dst_param.split([src_param.size(1), dst_param.size(1)-src_param.size(1)], dim=1)
#                         a1.set_(-src_param.data + a1.data)
#                         dst_param.set_(torch.cat([a1, a2], dim=1).data)
#                     else:  # 全连接层由2个维度分割
#                         a1, a2 = dst_param.split([src_param.size(0), dst_param.size(0)-src_param.size(0)], dim=0)
#                         a3, a4 = a1.split([src_param.size(1), dst_param.size(1)-src_param.size(1)], dim=1)
#                         a3.set_(-src_param.data + a3.data)
#                         a1.set_(torch.cat([a3, a4], dim=1).data)
#                         dst_param.set_(torch.cat([a1, a2], dim=0).data)
#                 else:  # 池化层由2个维度分割
#                     a1, a2 = dst_param.split([src_param.size(0), dst_param.size(0)-src_param.size(0)], dim=0)
#                     if dst_param.size(1) > src_param.size(1):
#                         a3, a4 = a1.split([src_param.size(1), dst_param.size(1)-src_param.size(1)], dim=1)
#                         a3.set_(-src_param.data + a3.data)
#                         a1.set_(torch.cat([a3, a4], dim=1).data)
#                         dst_param.set_(torch.cat([a1, a2], dim=0).data)
#                     else:
#                         a1.set_(-src_param.data + a1.data)
#                         dst_param.set_(torch.cat([a1, a2], dim=0).data)
#     return dst_model


# def adjust_model_params(source_params, target_width, model_type, dataset_type):
#     """
#         将源模型参数适配到目标模型宽度
#         Args:
#             source_params: 源模型参数(state_dict)
#             target_width: 目标模型宽度
#             model_type: 模型类型
#             dataset_type: 数据集类型
#         Returns:
#             target_state: 目标模型参数(state_dict)
#         """
#     # 创建目标宽度模型
#     target_model = create_model_instance(model_type, dataset_type, target_width)
#     target_state = target_model.state_dict()
    
#     for name, param in target_state.items():
#         if name not in source_params:
#             continue
            
#         src_param = source_params[name].data
        
#         # 按层类型处理
#         if 'features' in name:
#             layer_idx = int(name.split('.')[1])
#             # 根据层位置确定缩放比例
#             if layer_idx in [0]:  # conv1
#                 scale = target_width
#             elif layer_idx in [3]:  # conv2
#                 scale = target_width * target_width  # 输入和输出通道都缩放
#             elif layer_idx in [6, 8, 10]:  # 后续卷积层
#                 scale = target_width ** 2
#             else:
#                 scale = 1.0
#         elif 'classifier' in name:
#             scale = target_width ** 2  # 全连接层缩放
#         else:
#             scale = 1.0
            
#         # 参数截取逻辑
#         if param.dim() >= 2:
#             # 处理权重：按输出通道和输入通道截取
#             out_dim = int(param.size(0) * scale)
#             in_dim = int(param.size(1) * (target_width if 'features.0' in name else 1))
            
#             if src_param.dim() == 4:  # 卷积核
#                 target_state[name].data[:out_dim, :in_dim, :, :] = src_param[:out_dim, :in_dim, :, :]
#             else:  # 全连接层
#                 target_state[name].data[:out_dim, :in_dim] = src_param[:out_dim, :in_dim]
#         else:  # 偏置项
#             target_state[name].data[:int(param.size(0)*scale)] = src_param[:int(param.size(0)*scale)]
                
#     return target_state