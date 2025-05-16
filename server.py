# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # server所处的gpu显存剩余越多会对提升server运行速率有好处

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
from utils.custom_models import create_model_instance
from FLamingo.core.server import *
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, EMNIST
from torch.utils.data import DataLoader
# from algorithms import broadcast, aggregate
from collections import defaultdict
from utils.model_utils import *

class ProxClientInfo(ClientInfo):
    def __init__(self, rank):
        super().__init__(rank)
        self.width = 1.0  # 添加宽度属性
        self.state_dict = None
        self.datasize = 1
        self.model = None
        self.tau = None  # 本地迭代次数
        self.tcp = 0.0  # 本地计算时间
        self.tcm = 0.0
        self.acc_before_train = 0.0
        self.loss_before_train = 0.0
        self.acc_after_train = 0.0
        self.loss_after_train = 0.0
        self.local_iters = 50  # 本地迭代次数

class ProxServer(Server):
    def init(self):
        # self.model_size = []
        # for width in self.widths:
        #     model = create_model_instance(self.model_type, self.dataset_type, width)
        #     model_size = calculate_model_size(model)
        #     self.model_size.append(model_size)
        self.model_kwargs = {
            'model_type': self.model_type,
            'dataset_type': self.dataset_type
        }
        self.model_dict = {}
        self.widths = sorted(self.args.widths)  # 支持的模型宽度
        print(self.widths)
        self.num_params = {}
        self.model = create_model_instance(self.model_type, self.dataset_type, 1.0).to(self.device)  # 创建全局模型实例 
        for width in self.widths:
            cur_model = slice_model(self.model, width, self.device, **self.model_kwargs)
            self.model_dict[width] = cur_model
            # self.print_model_info(cur_model)
            self.num_params[width] = sum(p.numel() for p in cur_model.parameters())
        self.model.to(self.device)
        self.network = NetworkHandler()
        self.loss_func = nn.CrossEntropyLoss() if self.dataset_type in ['mnist', 'cifar10'] else nn.BCEWithLogitsLoss()
        self.width_assignment_strategy = self.args.width_assignment_strategy

        # 新增FedSNN相关参数
        self.cluster_num = int(self.num_clients / len(self.widths))
        self.cluster_centers = [None] * self.cluster_num
        self.max_cluster_iter = self.args.max_cluster_iter
        self.tau = {rank: self.args.local_iters for rank in range(1, self.num_clients+1)}  # 保存客户端tau值
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
        # self.log(f"stragglers: {self.stragglers_num}")
        # self.log(f"stragglers_idxes: {self.stragglers_idxes}")
        # self.log(f"non_stragglers_idxes: {self.non_stragglers_idxes}")
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

    def cluster_clients(self):
        """
        Cluster clients based on their data size.
        算法流程:
        Args:
            clients: 客户端列表
            model_type: server模型类型
            dataset_type: server数据集类型
        """
        cluster_centers = []
        max_clusters = self.num_clients // len(self.widths)
        for client in self.all_clients:
            if client.width == 1.0:
                center_model = create_model_instance(self.model_type,
                                                        self.dataset_type,
                                                        width=1.0).to(self.device)
                center_model.load_state_dict(client.state_dict)
                cluster_centers.append(center_model)
                if len(cluster_centers) >= max_clusters:
                    break
        
        self.num_clusters = len(cluster_centers)
        self.max_iterations = self.args.max_cluster_iter
        assignments = [-1] * self.num_clients
        converged = False
        iteration = 0

        self.log(f"Starting clustering with {self.num_clusters} clusters...")

        while not converged and iteration < self.max_iterations:
            self.log(f"Clustering iteration {iteration + 1}")
            old_assignments = assignments[:]

            # 计算相似度矩阵
            similarity_matrix = torch.zeros(self.num_clusters, self.num_clients, device=self.device)
            for c_idx, center_model in enumerate(cluster_centers):
                for cli_idx, client in enumerate(self.all_clients):
                    client_model = create_model_instance(self.model_type,
                                                         self.dataset_type,
                                                         width=client.width)
                    client_model.load_state_dict(client.state_dict)
                    # print(client_model)
                    client_model = client_model.to(self.device)
                    client_width = client.width
                    client_params_flat = flatten_parameters(client_model)
                    # 切片中心模型
                    sliced_center_model = slice_model(center_model, client_width, self.device, **self.model_kwargs)
                    sliced_center_params_flat = flatten_parameters(sliced_center_model)
                    # 计算相似度
                    similarity = torch.cosine_similarity(client_params_flat, sliced_center_params_flat, dim=0)
                    similarity_matrix[c_idx, cli_idx] = similarity

            # 2. 分配客户端 (两阶段)
            assignments = [-1] * self.num_clients # 重置分配
            unassigned_clients_indices = list(range(self.num_clients))
            assigned_count_this_iter = [0] * self.num_clusters # 追踪本轮分配情况

            # --- 第一阶段：优先分配给未满聚类 (Lines 6-9 in Alg 1, modified) ---
            clients_assigned_in_priority = 0
            temp_unassigned = [] # 存储第一阶段暂时无法分配的
            for cli_idx in unassigned_clients_indices:
                eligible_clusters = [] # 找到当前客户端可以分配的未满聚类
                for c_idx in range(self.num_clusters):
                    # 检查聚类是否未满 (基于本轮已分配的数量)
                    if assigned_count_this_iter[c_idx] < len(self.widths):
                        eligible_clusters.append(c_idx)

                if not eligible_clusters:
                    # 如果所有聚类都满了，该客户端暂时不分配，留给第二阶段
                    temp_unassigned.append(cli_idx)
                    continue

                # 在未满的聚类中找到最相似的
                best_cluster_idx = -1
                max_similarity = -float('inf')
                for c_idx in eligible_clusters:
                    if similarity_matrix[c_idx, cli_idx] > max_similarity:
                        max_similarity = similarity_matrix[c_idx, cli_idx]
                        best_cluster_idx = c_idx

                # if best_cluster_idx != -1:
                assignments[cli_idx] = best_cluster_idx
                assigned_count_this_iter[best_cluster_idx] += 1
                clients_assigned_in_priority += 1
                # else:
                #     # 理论上如果 eligible_clusters 不为空，总能找到一个 best_cluster_idx
                #     temp_unassigned.append(cli_idx) # 以防万一

            self.log(f"  Priority assignment phase: Assigned {clients_assigned_in_priority} clients.")
            unassigned_clients_indices = temp_unassigned # 更新未分配列表

            # --- 第二阶段：分配剩余客户端给最近的聚类 (Lines 8-9 in Alg 1, standard assignment) ---
            clients_assigned_in_second_phase = 0
            if unassigned_clients_indices:
                self.log(f"  Second phase assignment: Assigning {len(unassigned_clients_indices)} remaining clients.")
                for cli_idx in unassigned_clients_indices:
                    # 直接从所有聚类中找最相似的
                    best_cluster_idx = torch.argmax(similarity_matrix[:, cli_idx]).item()
                    assignments[cli_idx] = best_cluster_idx
                    # 注意：这里不再检查聚类大小限制，允许超出
                    assigned_count_this_iter[best_cluster_idx] += 1 # 继续追踪总分配数
                    clients_assigned_in_second_phase += 1
            self.log(f"  current {assignments}")
            # 检查是否收敛
            if assignments == old_assignments:
                converged = True
                self.log("Clustering converged.")
            else:
                # 3. 更新聚类中心 (Line 13 in Alg 1)
                new_cluster_centers = []
                for c_idx in range(self.num_clusters):
                    assigned_clients_indices = [idx for idx, cluster_id in enumerate(assignments) if cluster_id == c_idx]
                    if not assigned_clients_indices:
                        self.log(f"Warning: Cluster {c_idx} became empty. Re-using old center.")
                        new_cluster_centers.append(cluster_centers[c_idx])
                        continue
                    assigned_clients = [self.all_clients[cli_idx] for cli_idx in assigned_clients_indices]
                    
                    new_center = model_aggregation(assigned_clients, self.widths, 
                                                   cluster_centers[c_idx], False, self.device)
                    new_cluster_centers.append(new_center)
                cluster_centers = new_cluster_centers
            iteration += 1
        
        if not converged:   
            self.log(f"Clustering did not converge after {self.max_iterations} iterations.")

        # 构建最终聚类结果
        final_clusters = [[] for _ in range(self.num_clusters)]
        for cli_idx, cluster_id in enumerate(assignments):
            if cluster_id != -1:
                final_clusters[cluster_id].append(self.all_clients[cli_idx])

        return final_clusters
    
    def estimate_time(self, client, iters=30):
        # 已知每个客户端的computation和communication，估计客户端的用时
        ratio = client.num_params / self.num_params[1.0]
        return client.tcp * ratio * iters + client.tcm * 2 * ratio
    
    def FedSNN(self):
        cluster_results = self.cluster_clients()
        # 每个聚类内部分配 width
        for cluster in cluster_results:
            # 根据设备能力排序
            print(f"cur cluster {cluster}")
            cluster.sort(key=lambda x: x.tcm+x.tcp)
            # 从最慢的开始设置最小的宽度
            for i, width in enumerate(self.widths):
                cluster[i].width = width
                cluster[i].num_params = self.num_params[cluster[i].width]
                print(cluster[i].rank, cluster[i].width)
            if len(cluster) > len(self.widths):
                # 多出来的随机分配宽度
                print("additional")
                for j in range(len(self.widths), len(cluster)):
                    cluster[j].width = random.choice(self.widths)
                    cluster[j].num_params = self.num_params[cluster[j].width]
                    print(cluster[j].rank, cluster[j].width)
        # 估计所有客户端中用时最长的那个
        if self.use_adp_localiter:
            max_time = 0.0
            for client in self.all_clients:
                time = self.estimate_time(client)
                if time > max_time:
                    max_time = time
                # print(f"max_time: {max_time}')")
            for client in self.all_clients:
                ratio = client.num_params / self.num_params[client.width]
                est_cp_time = max_time - client.tcm * 2 * ratio
                client.local_iters = int(est_cp_time / (client.tcp * ratio))
                client.local_iters = max(min(client.local_iters, self.clip_iter['max']), self.clip_iter['min'])

    def multi_width_broadcast(self):
        """
        按客户端宽度生成兼容参数并广播
        """
        for client in self.selected_clients:
            client.state_dict = slice_model(self.model, client.width, self.device, **self.model_kwargs).state_dict()

        self.personalized_broadcast(
            common_data={'status': 'TRAINING'},
            personalized_attr=['state_dict', 'width',  'local_iters', 'tcp', 'tcm'],
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
        # self.assign_widths()

        # 初始化随机分配宽度
        for i in range(1, self.num_clients+1):
            client = self.get_client_by_rank(i)
            client.width = self.widths[i % len(self.widths)]
            client.state_dict = self.model_dict[client.width].state_dict()
            client.num_params = self.num_params[client.width]
            client.local_iters = self.local_iters
            # tcp, tcm
            client.tcp = self.sys_het_list[(i-1) % len(self.sys_het_list)]['computation']/10
            client.tcm = self.sys_het_list[(i-1) % len(self.sys_het_list)]['communication']
            self.log(f'client {i} width: {client.width}')

        self.generate_global_test_set()
        self.select_clients()
        # print("INIT ROUND")
        # self.personalized_broadcast(
        #     common_data={'status': 'INIT'},
        #     personalized_attr=['state_dict', 'width', 'tau']
        # )
        while True:
            print(f"Current global round: {self.global_round}.")
            # self.get_and_set_stragglers()

            if self.USE_CLUSTER:
                self.FedSNN()

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


def flatten_parameters(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

if __name__ == "__main__":
    server = ProxServer()
    server.run()