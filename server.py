# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 `utils` 目录到模块搜索路径
sys.path.append(os.path.join(current_dir, "..", "utils"))

import copy
# Now import FLamingo
from FLamingo.core.server import *
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, EMNIST
from torch.utils.data import DataLoader
# from algorithms import broadcast, aggregate
from collections import defaultdict
from utils.model_utils import *
from utils.custom_models import create_model_instance

class ProxClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        self.width = 1.0  # 添加宽度属性
        self.state_dict = None
        self.datasize = 1
        self.model = None

class ProxServer(Server):
    def init(self):
        self.model_dict = {}
        self.model_kwargs = {
            'model_type': self.model_type,
            'dataset_type': self.dataset_type
        }
        self.widths = sorted(self.args.widths)  # 支持的模型宽度
        print(self.widths)
        self.model = create_model_instance(self.model_type, self.dataset_type, 1.0).to(self.device)  # 创建全局模型实例 
        for width in self.widths:
            cur_model = slice_model(self.model, width, self.device, **self.model_kwargs)
            self.model_dict[width] = cur_model
        self.model.to(self.device)
        self.network = NetworkHandler()
        self.loss_func = nn.CrossEntropyLoss() if self.dataset_type in ['mnist', 'cifar10'] else nn.BCEWithLogitsLoss()
        self.width_assignment_strategy = self.args.width_assignment_strategy
        

    # def generate_global_test_set(self):
    #     """
    #     Generate a global test set.
    #     """
    #     if self.dataset_type == 'mnist':
    #         self.test_set = MNIST(root='../datasets', train=False, download=True, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,))
    #         ]))
    #     elif self.dataset_type == 'cifar10':
    #         self.test_set = CIFAR10(root='../datasets', train=False, download=True, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ]))
    #     elif self.dataset_type == 'emnist':
    #         self.test_set = EMNIST(root='../datasets', split='balanced', train=False, download=True, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,))
    #         ]))
    #     self.test_loader = DataLoader(
    #         dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False)
        
    # def test(self, model, test_loader):
    #     """
    #     Test the model.
    #     """
    #     model.eval()
    #     test_loss = 0.0
    #     correct = 0
    #     num_samples = 0
    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             data, target = data.to(self.device), target.to(self.device)
    #             output = model(data)
    #             test_loss += self.loss_func(output, target).item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #             num_samples += len(target)
    #     test_loss /= len(test_loader.dataset)
    #     accuracy = 100. * correct / len(test_loader.dataset)
    #     self.log(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    #     return {'test_loss': test_loss, 'test_acc': accuracy, 'test_samples': num_samples}
    
    # def average_client_info(self, client_list):
    #     """
    #     Average client info and log it
    #     """
    #     length = len(client_list)
    #     clients = [self.get_client_by_rank(rank) for rank in client_list]
    #     avg_train_loss = 0.0
    #     avg_test_loss, avg_test_acc = 0.0, 0.0
    #     avg_bf_test_acc, avg_bf_test_loss = 0.0, 0.0
    #     for client in clients:
    #         avg_train_loss += client.train_loss
    #         avg_test_acc += client.test_acc
    #         avg_test_loss += client.test_loss
    #         avg_bf_test_acc += client.bf_acc 
    #         avg_bf_test_loss += client.bf_loss
    #     self.log(f"Avg global info:\ntrain loss {avg_train_loss/length}, \
    #             \ntest acc {avg_test_acc/length}, \
    #             \ntest loss {avg_test_loss/length},\
    #             \navg_bf_test_acc {avg_bf_test_acc/length}, \
    #             \navg_bf_test_loss {avg_bf_test_loss/length} ")
        
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
            # self.log(f"Client {rank} gets {self.get_client_by_rank(rank).local_epochs} epochs and width {self.get_client_by_rank(rank).width}")
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
            for client in self.all_clients:
                client.width = random.choice(self.args.widths)
                client.state_dict = self.model_dict[client.width].state_dict()
                self.log(f"1st round, client {client.rank} assigned width: {client.width}")
            return 
        elif self.width_assignment_strategy == 'ordered':
            for i in range(1, self.num_clients+1):
                client = self.get_client_by_rank(i)
                client.width = self.widths[i % len(self.widths)]
                client.state_dict = self.model_dict[client.width].state_dict()
                self.log(f'client {i} width: {client.width}')
            return
        elif self.width_assignment_strategy == 'fixed':
            for rank in self.selected_clients_idxes:
                client = self.get_client_by_rank(rank)
                client.width = self.widths[0]
                client.state_dict = self.model_dict[client.width].state_dict()
                self.log(f"1st round, client {client.rank} assigned width: {client.width}")
            return
        elif self.width_assignment_strategy == 'cluster':
            clusters = self.cluster_clients(self.all_clients)
            # 按簇分配
            for cluster_id, clustered_clients in clusters.items():
                # 每个簇对应一个宽度
                width = self.widths[cluster_id % len(self.widths)]
                for client in clustered_clients:
                    client.width = width
                    client.state_dict = self.model_dict[client.width].state_dict()
                    self.log(f"1st round, client {client.rank} assigned width: {client.width}")
            return
        else:
            raise ValueError(f"Unknown width assignment strategy: {self.width_assignment_strategy}")
        
    def cluster_clients(self, clients):
        """
        Cluster clients based on their data size.
        """
        clusters = defaultdict(list)
        for client in clients:
            clusters[client.datasize].append(client)
        return clusters

    def multi_width_broadcast(self):
        """
        按客户端宽度生成兼容参数并广播
        """
        # dest_ranks = {''}
        for client in self.selected_clients:
            # 动态生成与客户端宽度兼容的参数字典
            # compatible_params = self.extract_width_params(self.model.state_dict(), client.width)
            compatible_model = slice_model(
                self.model, 
                client.width, 
                self.device,
                self.model_type, 
                self.dataset_type
            )
            compatible_params = compatible_model.state_dict()  # ✅ 适配参数
            data_to_send = {
                'status': 'TRAINING',
                'params': compatible_params,
                'straggler': client.is_straggler,
                'state_dict': client.state_dict,
                'global_round': self.global_round,
            }
            self.network.send(data=data_to_send, dest_rank = client.rank)
            # print(f"Server sending data to client {client.rank}: {data_to_send}")  # 添加调试信息
            print(f"Server broadcast to client {client.rank} succeed")
            # dest_ranks.add(client.rank)
        if self.verb:
            dest_ranks = self.selected_clients_idxes
            self.log(f'Server broadcast to {dest_ranks} succeed')

    def multi_width_aggregate(self, clients=None):
        """使用新的环形聚合策略"""
        
        # 准备聚合参数
        client_widths = [c.width for c in clients]
        global_model_copy = copy.deepcopy(self.model)
        
        # 执行新聚合
        self.model = model_aggregation(
            clients=clients,
            client_widths_input=client_widths,
            global_model=global_model_copy
        )
        

        # 保持原有逻辑：保存参数供聚类使用
        for client in self.selected_clients:
            client.last_params = copy.deepcopy(client.params)

    def run(self):
        """
        Basically init client, 
        select, broadcast, listen, aggregate 
        and wait for next round
        """
        self.init_clients(clientObj=ProxClientInfo)
        self.assign_widths()
        self.generate_global_test_set()
        self.select_clients()
        self.personalized_broadcast(
            common_data={'status': 'INIT'},
            personalized_attr=['state_dict', 'width']
        )
        while True:
            print(f"Current global round: {self.global_round}.")
            self.get_and_set_stragglers()

            for client in self.selected_clients:
                client.state_dict = slice_model(self.model, client.width, self.device, **self.model_kwargs).state_dict()

            self.personalized_broadcast(
                common_data={'status': 'TRAINING'},
                personalized_attr=['state_dict', 'is_straggler'],
            )

            print("Start calling listen()...")
            self.listen()
            print("End calling listen().")

            print("Start aggregating...")
            # self.multi_width_aggregate(clients=self.selected_clients)
            self.model = model_aggregation(
                self.selected_clients,
                self.widths,
                self.model
            )
            print("End aggregating.")
            
            print("Start evaluating...")
            # self.test(self.model, self.test_loader)
            for width in self.widths:
                test_model = slice_model(self.model, width, self.device, self.model_type, self.dataset_type)
                test_dic = self.test(test_model, self.test_loader)
                self.log(f'width:{width}=====test acc: {test_dic["test_acc"]}, test_loss: {test_dic["test_loss"]}')
            print("End evaluating.")

            self.average_client_info(self.selected_clients_idxes)

            self.finalize_round()
            print(f"global_round {self.global_round-1} ends.")
            if self.global_round >= self.max_epochs:
                print("Reaching epochs limit.")
                break
        # out of loop
        self.log("Yes, just end your job")
        self.stop_all()


if __name__ == "__main__":
    server = ProxServer()
    server.run()