# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
import copy
# Now import FLamingo
from FLamingo.core.server import *
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
# from algorithms import broadcast, aggregate
from collections import defaultdict
from model_utils import create_model_instance

class ProxClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        self.bf_acc = 0.0
        self.bf_loss = 0.0
        self.is_straggler = False
        self.local_epochs = 10
        self.width = 1.0  # 添加宽度属性

class ProxServer(Server):
    def init(self):
        # TODO: 你想写的东西放在这里
        # self.model =
        # self.dataset =
        
        self.widths = self.args.widths  # 支持的模型宽度
        self.width_assignment_strategy = self.args.width_assignment_strategy
        self.model_widths = {}  # 用于存储不同宽度的模型

    def generate_global_test_set(self):
        """
        Generate a global test set.
        """
        if self.dataset_type == 'mnist':
            self.test_set = MNIST(root='../datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        elif self.dataset_type == 'cifar10':
            self.test_set = CIFAR10(root='../datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        self.test_loader = DataLoader(
            dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False)
        
    def test(self, model, test_loader):
        """
        Test the model.
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        self.log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
        return {'test_loss': test_loss, 'accuracy': accuracy}
    
    def average_client_info(self, client_list):
        """
        Average client info and log it
        """
        length = len(client_list)
        clients = [self.get_client_by_rank(rank) for rank in client_list]
        avg_train_loss = 0.0
        avg_test_loss, avg_test_acc = 0.0, 0.0
        avg_bf_test_acc, avg_bf_test_loss = 0.0, 0.0
        for client in clients:
            avg_train_loss += client.train_loss
            avg_test_acc += client.test_acc
            avg_test_loss += client.test_loss
            avg_bf_test_acc += client.bf_acc 
            avg_bf_test_loss += client.bf_loss
        self.log(f"Avg global info:\ntrain loss {avg_train_loss/length}, \
                \ntest acc {avg_test_acc/length}, \
                \ntest loss {avg_test_loss/length},\
                \navg_bf_test_acc {avg_bf_test_acc/length}, \
                \navg_bf_test_loss {avg_bf_test_loss/length} ")
        
    def get_and_set_stragglers(self):
        """
        Generate a list of stagglers according to config.stragglers.
        """
        self.stragglers_num = int(self.stragglers * self.num_training_clients)
        # print(self.stragglers_num)
        self.stragglers_idxes = random.sample(self.selected_clients_idxes, k=self.stragglers_num)
        # print(self.stragglers_idxes, self.size)
        self.non_stragglers_idxes = list(set(self.selected_clients_idxes) - set(self.stragglers_idxes))
        for rank in self.stragglers_idxes:
            self.get_client_by_rank(rank).is_straggler = True
            self.get_client_by_rank(rank).local_epochs = random.choice(range(1, self.local_epochs+1))
            self.log(f"Client {rank} gets {self.get_client_by_rank(rank).local_epochs} epochs and width {self.get_client_by_rank(rank).width}")
        for rank in self.non_stragglers_idxes:
            self.get_client_by_rank(rank).is_straggler = False
            self.get_client_by_rank(rank).local_epochs = self.local_epochs
        self.log(f"stragglers: {self.stragglers_num}")
        self.log(f"stragglers_idxes: {self.stragglers_idxes}")
        self.log(f"non_stragglers_idxes: {self.non_stragglers_idxes}")
        return self.stragglers_idxes, self.non_stragglers_idxes
    
    # 根据策略为客户端分配宽度
    def assign_widths(self):
        """
        Assign width to each client,
        based on strategy.
        """
        if self.width_assignment_strategy == 'random':
            for rank in self.selected_clients_idxes:
                client = self.get_client_by_rank(rank)
                client.width = random.choice(self.widths)
        elif self.width_assignment_strategy == 'ordered':
            # Here to complete your own strategy
            for rank in self.selected_clients_idxes:
                client = self.get_client_by_rank(rank)
                client.width = self.widths[rank%len(self.widths)]
            # pass
        elif self.width_assignment_strategy == 'fixed':
            for rank in self.selected_clients_idxes:
                client = self.get_client_by_rank(rank)
                client.width = self.widths[0]
        else:
            raise ValueError(f"Unknown width assignment strategy: {self.width_assignment_strategy}")

    def multi_width_broadcast(self):
        """
        按客户端宽度生成兼容参数并广播
        """
        dest_ranks = {None}
        for client in self.selected_clients:
            # 动态生成与客户端宽度兼容的参数字典
            compatible_params = self.generate_compatible_params(client.width)
            self.network.send(
                data = {
                    'status': 'TRAINING',
                    'params': compatible_params,
                    'width': client.width,
                    'local_epochs': self.local_epochs,
                    'straggler': client.is_straggler,
                    'global_round': self.global_round
                },
                dest_rank = client.rank
            )
            print(f"Server broadcast to client {client.rank} succeed")
            dest_ranks.add(client.rank)
        if self.verb:
            self.log(f'Server broadcast to {dest_ranks} succeed')

    def generate_compatible_params(self, width):
        """
        生成与指定宽度兼容的参数字典
        Args:
            width: 客户端的宽度client.width
        """
        global_width = 1.0  # 全局模型的宽度默认为1.0
        width_ratio = width / global_width
        
        # 获取全局模型的完整参数
        global_state = self.model.state_dict()
        
        # 创建目标宽度模型
        model_subset = create_model_instance(self.model_type, self.dataset_type, width)
        subset_state = model_subset.state_dict()
        
        # 按宽度比例裁剪参数
        for name, param in global_state.items():
            if name not in subset_state:
                continue  # 跳过不存在的层
            
            ## TODO:
            # for target_param, global_param in zip(slim_model.parameters(), fat_model.parameters()):
            #     target_param = global_param[:len(target_param)]
                
            
            # 按层类型裁剪
            if 'conv' in name and 'weight' in name:
                # 卷积核权重 [out_channels, in_channels, kH, kW]
                out_channels = int(param.shape[0] * width_ratio)
                in_channels = int(param.shape[1] * width_ratio)
                subset_state[name] = param[:out_channels, :in_channels, ...]
                
            elif 'conv' in name and 'bias' in name:
                out_channels = int(param.shape[0] * width_ratio)
                subset_state[name] = param[:out_channels]
                
            elif 'bn' in name:
                # BN层参数（weight/bias/running_mean/running_var）
                channels = int(param.shape[0] * width_ratio)
                subset_state[name] = param[:channels]
                
            else:
                # 全连接层等不兼容层直接跳过
                continue
        
        return subset_state

    def average_group_params(self, clients, weight_by_samples=True):
        """
        对同一宽度组内的客户端参数按样本量加权平均
        Args:
            clients: 客户端列表(需包含 `params` 和 `train_samples` 属性)
            # weight_by_samples: 是否按样本量加权(否则按客户端数量平均)
        Returns:
            avg_params: 平均后的参数字典(state_dict)
        """
        if not clients:
            return {}
        
        # 初始化累加器&权重和
        avg_params = {}
        total_weight = 0.0

        # 处理client的参数
        for client in clients:
            weight = client.train_samples if weight_by_samples else 1.0
            total_weight += weight

            for name, param in client.params.items():
                # param = param.to('cpu')  # 为了节省显存, 如果报错可以省略
                if name not in avg_params:
                    avg_params[name] = torch.zeros_like(param)*weight
                else:
                    avg_params[name] += param*weight
        
        # 计算加权平均
        for name in avg_params:
            avg_params[name] /= total_weight

        return avg_params

    def crop_params(self, source_params, source_width, target_width, model_type):
        """
        将源宽度参数裁剪为目标宽度兼容格式
        Args:
            source_params: 源参数字典(state_dict)
            source_width: 源模型宽度(如0.5)
            target_width: 目标模型宽度(如1.0)
            model_type: 模型类型(用于确定裁剪规则, 暂未用到)
        Returns:
            cropped_params: 裁剪后的参数字典(state_dict)'
        """
        def _get_dim(original_dim, width_ratio):
            # 计算裁剪维度
            return int(original_dim*width_ratio+0.5)  # 四舍五入

        cropped_params = {}
        width_ratio = target_width / source_width  # 计算扩展比例, 例如: 目标1.0 / 源0.5 = 2.0(需要扩展2倍)
        
        # 处理过程
        for name, param in source_params.items():
            # 全连接层无需处理
            if 'classifier' in name or 'fc' in name:
                continue
            
            # 按层类型填充&裁剪
            # 处理卷积层权重
            if 'conv' in name and 'weight' in name:
                # PyTorch固定的卷积核权重格式: [out_channels, in_channels, kH, kW]
                # kH == kernel_height, kW == kernel_width
                original_out = param.size(0)
                original_in = param.size(1)
                new_out = _get_dim(original_out, width_ratio)
                new_in = _get_dim(original_in, width_ratio)

                # 填充&裁剪(一般target_width>source_width, 需要填充)
                if width_ratio >= 1.0:  # 填充
                    # 创建一个填充了零的新张量, 并将原始参数复制到其中
                    padded_param = torch.zeros((new_out, new_in, *param.shape[2:]), dtype = param.dtype)
                    padded_param[:original_out, :original_in, ...] = param
                    cropped_params[name] = padded_param
                else:  #裁剪
                    cropped_params = param[:new_out, :new_in, ...]
            # 处理卷积层偏置
            elif 'conv' in name and 'bias' in name:
                # PyTorch固定的卷积偏置格式: [out_channels]
                new_dim = _get_dim(param.size(0), width_ratio)
                # 如果目标宽度小于或等于源宽度-->直接裁剪偏置参数
                # 如果目标宽度大于源宽度-->将原始偏置参数与零张量拼接
                cropped_params[name] = param[:new_dim] if width_ratio <= 1.0 else torch.cat([param, torch.zeros(new_dim - param.size(0))])
            # 处理bn层
            elif 'bn' in name:
                # 按通道数裁剪
                new_dim = _get_dim(param.size(0), width_ratio)
                cropped_params[name] = param[:new_dim]

        return cropped_params

    def multi_width_aggregate(self, clients=None):
        """
        修改内容: 按区域逐层聚合
        Args:
            clients: 需要处理的客户端集(list)
        Returns:
            None(用load_state_dict()方法为模型载入参数)
        """
        # 初始化全局模型
        global_state = self.model.state_dict()
        aggregated_params = copy.deepcopy(global_state)

        # 1. 按宽度分组, 遍历每一个宽度
        for width in self.widths:
            # eligible_clients的定义代表共有此width的客户端
            # len(eligible_clients)代表共有此width的客户端数量
            eligible_clients = [client for client in clients if client.width >= width]
            if not eligible_clients:
                continue
            # 计算当前宽度组的增量参数
            group_avg_params = self.average_group_params(eligible_clients)
            # 裁剪当前宽度的参数
            cropped_params = self.crop_params(group_avg_params, width-0.25, width)

            # 2. 按宽度逐层合并
            for name, param in cropped_params.items():
                if name in aggregated_params:
                    if param.shape == aggregated_params[name].shape:
                        weight = len(eligible_clients)/len(clients)
                        aggregated_params[name] = param*weight
                    else: 
                        self.log(f"参数 {name} 形状不匹配: 全局 {aggregated_params[name].shape} 而当前 {param.shape}")
        
        self.model.load_state_dict(aggregated_params, strict = False)

    def run(self):
        """
        Basically init client, 
        select, broadcast, listen, aggregate 
        and wait for next round
        """
        # print(self.size,flush=True)
        self.init_clients(clientObj=ProxClientInfo)
        self.generate_global_test_set()
        # print(os.environ)
        # TODO: assign width
        for client in self.all_clients:
            client.width = random.choice(self.args.widths)
            # self.send(client)
            self.log(f"Client {client.rank} assigned width: {client.width}")
        while True:
            self.select_clients()
            self.get_and_set_stragglers()
            # self.assign_widths()
            msg_dict = {}

            """
            # ---------------straggler & non_straggler---------------
            # Non stragglers
            # 广播不同宽度的模型参数
            for rank in self.non_stragglers_idxes: # <--不可以这么写
            # 注意non_straggler这里的处理逻辑：dest_ranks是列表不是元素
                # print(f"About to send to non_straggler client {rank}", flush=True)
                client = self.get_client_by_rank(rank)
                msg_dict[rank] = {
                    'status': 'TRAINING',
                    'params': self.export_model_parameter(),
                    'local_epochs': self.local_epochs,
                    'straggler': False,
                    'global_round': self.global_round,
                    'width': client.width
                }
                # print(f"Finished sending to non_straggler client {rank}", flush=True)
            self.multi_width_broadcast(msg_dict)

            # stragglers
            for rank in self.stragglers_idxes:
                # print(f"About to send to straggler client {rank}", flush=True)
                client = self.get_client_by_rank(rank)
                self.network.send(
                    data={
                        'status': 'TRAINING',
                        'params': self.export_model_parameter(),
                        'local_epochs': client.local_epochs,
                        'straggler': True,
                        'global_round': self.global_round,
                        'width': client.width
                    },
                    dest_rank=rank
                )
                # print(f"Finished sending to straggler client {rank}", flush=True)
            # -------------------------end-------------------------
            """

            self.multi_width_broadcast()

            print("Before calling listen()")
            self.listen()
            print("After calling listen()")
            self.multi_width_aggregate(clients = self.selected_clients)
            self.test(self.model, self.test_loader)
            self.average_client_info(self.selected_clients_idxes)
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                break
        # out of loop
        self.log("Yes, just end your job")
        self.stop_all()


if __name__ == "__main__":
    server = ProxServer()
    server.run()