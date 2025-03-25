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

    def extract_width_params(self, params, width):
        """
        根据宽度提取差异部分
        Args:
            width: 当前模型宽度
        Return:
            sub_model: 裁剪的子模型params字典
        """
        sub_model = create_model_instance(self.model_type, self.dataset_type, width)
        sub_model.load_state_dict(params, strict=False)
        return sub_model.state_dict()

    def multi_width_broadcast(self):
        """
        按客户端宽度生成兼容参数并广播
        """
        dest_ranks = {None}
        for client in self.selected_clients:
            # 动态生成与客户端宽度兼容的参数字典
            compatible_params = self.extract_width_params(self.model.state_dict(), client.width)
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

    def multi_width_aggregate(self, clients=None):
        """
        按区域逐层聚合: 每个宽度区间视为一个层级, 例如0.0-0.25为一个层级, 0.25-0.5为一个层级, 以此类推
        Args:
            clients: 需要处理的客户端集(list)
        Returns:
            None(用load_state_dict()方法为模型载入参数)
        """
        # 初始化全局模型
        sorted_widths = sorted(self.args.widths)
        global_model = create_model_instance(self.model_type, self.dataset_type, 1.0)
        global_state = global_model.state_dict()

        # 初始化各个宽度层级对应的累加器
        layer_aggregates = {weight: {key: torch.zeros_like(value) for key, value in global_state.items()} for weight in sorted_widths}
        # layer_weights = {weight: 0.0 for weight in sorted_widths}

        # 按层级从低层到高层逐步聚合: [0.0, 0.25] --> [0.75, 1.0]
        # width_level: 每个层级最右端区间作为当前层级代表, 便于代码书写
        for idx, width_level in enumerate(sorted_widths):
            # 筛选共有当前层级的客户端eligible_clients
            eligible_clients = [client for client in clients if client.width>=width_level]
            total_weight = sum(client.train_samples for client in eligible_clients)
            if total_weight == 0:
                continue
            # 提取所有客户端当前层级的参数然后累加: 比如当前层级为0.5, 那么就提取模型[0.25, 0.5]之间的params
            for client in eligible_clients:
                # 第一个宽度层级直接提取该宽度下的参数
                if idx == 0:
                    delta_params = self.extract_width_params(client.params, width_level)
                # 后面的宽度层级需计算当前宽度与前一宽度的参数差异
                else:
                    prev_params = self.extract_width_params(client.params, sorted_widths[idx-1])
                    current_params = self.extract_width_params(client.params, width_level)
                    delta_params = {key: current_params[key]-prev_params.get(key, 0) for key in current_params}  # 差异计算               
                # 当前层级参数提取出来之后进行加权累加
                weight = client.train_samples/total_weight
                for name in delta_params:
                    if name in layer_aggregates[width_level]:
                        layer_aggregates[width_level][name] += delta_params[name]*weight
            
            # 全局模型更新
            current_layer_state = global_model.state_dict()
            for name in current_layer_state:
                if name in layer_aggregates[width_level]:
                    current_layer_state[name] += layer_aggregates[width_level][name]
            global_model.load_state_dict(current_layer_state)

        self.model.load_state_dict(global_model.state_dict())

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