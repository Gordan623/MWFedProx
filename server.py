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
        self.tau = None  # 本地迭代次数
        self.tcp = None  # 本地计算时间
        self.tcm = None

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

        # 新增FedSNN相关参数
        self.cluster_num = int(self.num_clients / len(self.widths))
        self.cluster_centers = [None] * self.cluster_num
        self.max_cluster_iter = self.args.max_cluster_iter
        self.tau = {rank: self.args.local_iteration for rank in range(1, self.num_clients+1)}  # 保存客户端tau值
        self.tcp_history = defaultdict(list)  # 保存客户端计算时间
        self.tcm_history = defaultdict(list)  # 保存客户端通信时间

        
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
            clusters = self.cluster_clients(self.all_clients, self.model_type, self.dataset_type)
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
    
    def create_model_and_load(self, client, model_type=None, dataset_type=None):
        """
        创建模型实例并加载参数
        Args:
            client: 客户端对象
        Returns:
            model: 创建的模型实例
        """
        model = create_model_instance(model_type, dataset_type, client.width).to(self.device)
        model.load_state_dict(client.state_dict)
        return model

    def flatten_paras(self, params):
        """
        将模型的参数展平为一维向量
        Args:
            params: 模型的参数列表
        Returns:
            flatten_params: 展平后的参数向量
        """
        flatten_params = []
        for p in params:
            flatten_params.append(torch.reshape(p.data.clone().detach(), [-1]))
        return torch.cat(flatten_params)

    def cluster_clients(self, clients, model_type, dataset_type):
        """
        Cluster clients based on their data size.
        算法流程:
        Args:
            clients: 客户端列表
            model_type: server模型类型
            dataset_type: server数据集类型
        """
        # 初始化聚类中心
        if not any(self.cluster_centers):
            count = 0
            for client in clients:
                if client.width == 1.0:
                    temp_model = self.create_model_and_load(client, model_type, dataset_type)
                    self.cluster_centers[count] = copy.deepcopy(temp_model)
                    count += 1
                    if count == self.cluster_num:
                        break
        
        # 创建每个ClientInfo对应的model
        client_models = [self.create_model_and_load(client, model_type, dataset_type) for client in clients]

        print("Start clustering clients...")
        # 迭代聚类
        for iter in range(self.max_cluster_iter):
            # 1. 初始化距离矩阵
            distance_matrix = torch.zeros((len(clients), self.cluster_num))

            # 2. 计算每个客户端与每个聚类中心的余弦相似度并更新距离矩阵
            for i, client_model in enumerate(client_models):
                current_client = clients[i]  # 获取当前客户端对象
                current_width = current_client.width  # 明确获取当前客户端宽度
                for j in range(self.cluster_num):
                    centroid_model = slice_model(self.cluster_centers[j], current_width, self.device, **self.model_kwargs)
                    # print(f"client_model: {client_model}\ncentroid_model: {centroid_model}")
                    distance_matrix[i][j] = torch.cosine_similarity(  # distance_matrix[i][j]表示第i个客户端与第j个聚簇中心的余弦相似度
                        self.flatten_paras(client_model.parameters()),
                        self.flatten_paras(centroid_model.parameters()),
                        dim=0
                    )
            print(distance_matrix)
            # 3. 分配设备到聚类
            cluster_assignments = torch.argmax(distance_matrix, dim=1)  # 每个客户端分配到的聚类

            # 4. 更新聚类中心
            new_centers = []
            for j in range(self.cluster_num):
                # 获取该簇的客户端索引
                cluster_indices = np.where(cluster_assignments == j)[0]
                if len(cluster_indices) == 0:
                    new_centers.append(self.cluster_centers[j])
                    continue
                    
                # 聚合模型（需要处理state_dict）
                aggregated = create_model_instance(model_type, dataset_type, 1.0).to(self.device)
                
                # 加权聚合逻辑
                total_weight = sum(clients[i].datasize for i in cluster_indices)
                for param in aggregated.parameters():
                    param.data.zero_()
                
                for i in cluster_indices:
                    client = clients[i]
                    weight = client.datasize / total_weight
                    # 必须从state_dict创建模型实例
                    client_model = self.create_model_and_load(client, model_type, dataset_type)
                    # 统一宽度后聚合
                    full_model = expand_model(client_model, 1.0, self.device, **self.model_kwargs)
                    for agg_param, cli_param in zip(aggregated.parameters(), full_model.parameters()):
                        agg_param.data += cli_param.data * weight
                        
                new_centers.append(aggregated)
            
            self.cluster_centers = new_centers
            print(f"iter {iter+1}(from 1 to 10) ends.")
        # print(f"Cluster centers = {self.cluster_centers}")



    def dynamic_tau_adjustment(self, clients):
        """动态 tau 调整策略，跳过缺失统计的客户端"""
        # 1. 计算最后三轮的平均值, 使用nanmean并允许 NaN 产生
        tcp = {}
        tcm = {}
        for c in clients:
            # 取最后 3 条记录，允许产生空切片
            last_tcp = self.tcp_history[c.rank][-3:]
            last_tcm = self.tcm_history[c.rank][-3:]
            # np.nanmean会忽略NaN, 但全空时仍返回NaN并警告: contentReference[oaicite:1]{index=1}
            tcp[c.rank] = np.nanmean(last_tcp)
            tcm[c.rank] = np.nanmean(last_tcm)

        # 2. 计算max_time, 只考虑有有效值的客户端
        valid_clients = [c for c in clients if not (np.isnan(tcp[c.rank]) or np.isnan(tcm[c.rank]))]
        if not valid_clients:
            # 如果一个也没有有效数据, 则直接返回, 不做任何调整
            return

        max_time = max(
            tcm[c.rank] + (10 / self.tau[c.rank]) * tcp[c.rank]
            for c in valid_clients
        )

        # 3. 对每个客户端进行tau调整, 忽略无效者
        for c in clients:
            if np.isnan(tcp[c.rank]) or np.isnan(tcm[c.rank]) or tcp[c.rank] == 0:
                # 跳过没有统计数据或tcp为 0 的客户端
                continue

            # 公式: new_tau = (max_time - tcm) * old_tau / (tcp/2)
            ratio = (max_time - tcm[c.rank]) * self.tau[c.rank] / (tcp[c.rank] / 2)
            # 确保ratio为实数且不为NaN
            if np.isfinite(ratio):
                # 取整并裁剪到合理区间
                self.tau[c.rank] = int(np.clip(ratio, 50, 200))
                c.tau = self.tau[c.rank]



    def multi_width_broadcast(self):
        """
        按客户端宽度生成兼容参数并广播
        """
        for client in self.selected_clients:
            client.state_dict = slice_model(self.model, client.width, self.device, **self.model_kwargs).state_dict()

        self.personalized_broadcast(
            common_data={'status': 'TRAINING'},
            personalized_attr=['state_dict', 'is_straggler', 'tau'],
            # personalized_attr=['state_dict', 'is_straggler'],
        )

        if self.verb:
            dest_ranks = self.selected_clients_idxes
            self.log(f'Server broadcast to {dest_ranks} succeed')

    def multi_width_aggregate(self, clients=None):
        """使用新的环形聚合策略"""
        
        # 执行新聚合
        self.model = model_aggregation(
            clients=clients,
            client_widths_input=self.widths,
            global_model=self.model
        )

    def evaluate(self):
        for width in self.widths:
            test_model = slice_model(self.model, width, self.device, self.model_type, self.dataset_type)
            test_dic = self.test(test_model, self.test_loader)
            self.log(f'width:{width}=====test acc: {test_dic["test_acc"]}, test_loss: {test_dic["test_loss"]}')

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
        print("INIT ROUND")
        self.personalized_broadcast(
            common_data={'status': 'INIT'},
            personalized_attr=['state_dict', 'width', 'tau']
        )
        while True:
            print(f"Current global round: {self.global_round}.")
            self.get_and_set_stragglers()

            # if self.global_round > 0:
            self.cluster_clients(self.selected_clients, self.model_type, self.dataset_type)
            self.dynamic_tau_adjustment(self.selected_clients)

            print("Start broadcasting...")
            self.multi_width_broadcast()
            print("End broadcasting.")

            print("Start calling listen()...")
            self.listen()
            print("End calling listen().")

            print("Start aggregating...")
            self.multi_width_aggregate(clients=self.selected_clients)
            print("End aggregating.")
            
            print("Start evaluating...")
            self.evaluate()
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