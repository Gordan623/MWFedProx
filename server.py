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

    # def cosine_similarity(self, params1, params2):  # torch.nn.functional.cosine_similarity只能用来计算展平向量
    #     # 计算两个模型参数的余弦相似度(跳过BN层)
    #     dot_product = 0
    #     norm1, norm2 = 0, 0
    #     for name in params1:
    #         if 'bn' not in name and name in params2:
    #             p1, p2 = params1[name], params2[name]
    #             dot_product += torch.dot(p1.flatten(), p2.flatten())
    #             norm1 += torch.norm(p1) ** 2
    #             norm2 += torch.norm(p2) ** 2
    #     return dot_product / (torch.sqrt(norm1) * torch.sqrt(norm2) + 1e-8)

    # def cluster_clients(self, clients):
    #     """
    #     聚簇逻辑
    #     Args:
    #         clients: 即将被分配的客户端集
    #     Return:
    #         clusters: 聚类完的簇
    #     """
    #     # 初始化: 从所有设备中随机选取 |S| 个设备(这里设为M), 并将这些设备的本地模型参数作为初始簇中心
    #     M = len(self.args.widths)  # 簇的数量
    #     active_clients = [client for client in clients if client.last_params is not None]
    #     if len(active_clients) < M:
    #         return defaultdict(list)  # 如果活跃客户端不足, 返回空簇

    #     # 随机选取 M 个客户端的参数作为初始簇中心
    #     centroids = [copy.deepcopy(active_clients[i].last_params) for i in range(M)]

    #     # 迭代更新簇
    #     max_iterations = 5  # 最大迭代次数
    #     for _ in range(max_iterations):
    #         clusters = defaultdict(list)
    #         # 簇分配: 计算每个设备与所有簇中心的余弦相似度, 将设备分配到相似度最高的簇
    #         for client in active_clients:
    #             max_sim, best_cluster = -1, 0
    #             for i, centroid in enumerate(centroids):
    #                 sim = self.cosine_similarity(client.last_params, centroid)
    #                 if sim > max_sim:
    #                     max_sim, best_cluster = sim, i
    #             clusters[best_cluster].append(client)

    #         # # TODO: 检查每个簇是否满足最小设备数量要求: 如果某个簇中设备数量还未达到预设的上限
    #         # # (例如，每个簇至少需要超过 M 个设备), 则优先将设备分配到该簇

    #         # 更新簇中心: 对于每个簇, 根据簇内所有设备上传的本地模型参数进行聚合(类似模型聚合), 更新簇中心
    #         new_centroids = []
    #         for cluster_id, clustered_clients in clusters.items():
    #             if clustered_clients:
    #                 avg_params = self.average_parameters([client.last_params for client in clustered_clients])
    #                 new_centroids.append(avg_params)
    #             else:
    #                 # 如果某个簇为空, 保持原中心
    #                 new_centroids.append(centroids[cluster_id])

    #         # 检查收敛条件
    #         if all(self.cosine_similarity(new_centroids[i], centroids[i]) > 0.999 for i in range(M)):
    #             break
    #         centroids = new_centroids

    #     # 存储并返回聚簇结果
    #     self.clusters = clusters
    #     return clusters

    # def average_parameters(self, params_list):
    #     """
    #     对多个模型参数进行加权平均
    #     Args:
    #         params_list: 模型参数列表
    #     Return:
    #         avg_params: 平均后的模型参数
    #     """
    #     avg_params = copy.deepcopy(params_list[0])
    #     for key in avg_params:
    #         avg_params[key] = torch.mean(torch.stack([params[key] for params in params_list]), dim=0)
    #     return avg_params

    # def extract_width_params(self, params, width):
    #     """
    #     根据宽度提取差异部分
    #     Args:
    #         width: 当前模型宽度
    #     Return:
    #         sub_model: 裁剪的子模型params字典
    #     """
    #     if width == 1.0 or not hasattr(self.model, 'width'):
    #         # 直接返回原始参数（针对不支持宽度的变体）
    #         return {k: v.clone() for k, v in params.items()}

    #     src_model = create_model_instance(self.model_type, self.dataset_type, 1.0)
    #     src_model = src_model.to(self.device)
    #     src_model.load_state_dict(params)
    #     sub_model = model_divided(src_model, width, self.model_type, self.dataset_type)
    #     # sub_model = sub_model.to(self.device)  # model_divided已经规定输出的model在GPU上了

    #     return sub_model.state_dict()

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
    
    """
    def multi_width_aggregate(self, device_updates, width_list, model_type, dataset_type):
        # 计算各宽度层的累计权重(train_samples的和)
        width_remmer = np.zeros(len(width_list), dtype=np.float32)
        for i, w in enumerate(width_list):
            for update in device_updates:
                if update['width'] >= w:
                    width_remmer[i] += update['train_samples']
        
        # 初始化每个宽度层级的模型累加器(全零模型, 使用全宽模型作为基底, 并用scale_model将参数置0)
        model_remmer = []
        for w in width_list:
            base_model = create_model_instance(model_type, dataset_type, 1.0)
            base_model = base_model.to(self.device)
            base_model = copy.deepcopy(scale_model(base_model, 0))   # scale_model(model, 0)将所有参数置0
            model_remmer.append(base_model)
        
        # 对每个宽度层级聚合各客户端更新
        for i, target_w in enumerate(width_list):
            for update in device_updates:
                if update['width'] >= target_w:
                    # 构造客户端模型(使用客户端 width)
                    client_model = create_model_instance(model_type, dataset_type, update['width'])
                    client_model = client_model.to(self.device)
                    client_model.load_state_dict(update['params'], strict=False)
                    # 计算权重(客户端样本占该层级总样本比例)
                    weight = float(update['train_samples']) / width_remmer[i]
                    # 先对客户端模型进行scaling, scale_model(model, factor)将参数乘以factor
                    scaled_client_model = scale_model(client_model, weight)
                    # 累加到当前层聚合器上, 用add_model_with_different_width对不同宽度的模型实现"加法"
                    model_remmer[i] = add_model_with_different_width(model_remmer[i], 1.0, scaled_client_model, update['width'])
        
        # 对不同宽度层级进行差分调整, 构造最终全局模型
        for i, target_w in enumerate(width_list):
            if i == 0:
                # 对最窄的层级, 使用model_divided得到该层更新
                base_zero = scale_model(create_model_instance(model_type, dataset_type, 1.0), 0)
                model_remmer[i] = add_model_with_different_width(
                    base_zero, 1.0, model_divided(model_remmer[i], target_w, model_type), target_w)
            elif i < len(width_list) - 1:
                base_zero = scale_model(create_model_instance(model_type, dataset_type, 1.0), 0)
                model_remmer[i] = add_model_with_different_width(
                    base_zero, 1.0, model_divided(model_remmer[i], target_w, model_type), target_w)
                model_remmer[i] = minus_model_with_differnt_width(
                    model_remmer[i], model_divided(model_remmer[i], width_list[i-1], model_type))
            else:
                model_remmer[i] = minus_model_with_differnt_width(
                    model_remmer[i], model_divided(model_remmer[i], width_list[i-1], model_type))
        
        # 将各层聚合结果累加, 得到最终全局模型
        final_model = model_remmer[0]
        for i in range(1, len(width_list)):
            final_model = add_model_with_different_width(final_model, 1.0, model_remmer[i], 1.0)
        
        return final_model
    """

    """
    def multi_width_aggregate(self, clients=None):

        # 按区域逐层聚合: 每个宽度区间视为一个层级, 例如0.0-0.25为一个层级, 0.25-0.5为一个层级, 以此类推
        # Args:
        #     clients: 需要处理的客户端集(list)
        # Returns:
        #     None(用load_state_dict()方法为模型载入参数)

         # 统计客户端的宽度分布
        width_counts = {}
        for client in clients:
            width = client.width
            if width not in width_counts:
                width_counts[width] = 0
            width_counts[width] += 1

        # 选择出现次数最多的宽度作为全局模型的初始宽度, 初始化全局模型
        most_common_width = max(width_counts, key=width_counts.get)
        global_model = create_model_instance(self.model_type, self.dataset_type, most_common_width)
        global_model = global_model.to(self.device)  # 确保全局模型在GPU上
        global_state = global_model.state_dict()
        sorted_widths = sorted(self.args.widths)


        # 初始化各个宽度层级对应的累加器
        layer_aggregates = {weight: {key: torch.zeros_like(value) for key, value in global_state.items()} for weight in sorted_widths}
        # layer_weights = {weight: 0.0 for weight in sorted_widths}

        # 获取聚簇结果
        clusters = self.clusters

        # 按层级从低层到高层逐步聚合: [0.0, 0.25] --> [0.75, 1.0]
        # width_level: 每个层级最右端区间作为当前层级代表, 便于代码书写
        for idx, width_level in enumerate(sorted_widths):
            # 筛选共有当前层级的客户端eligible_clients
            eligible_clients = [client for client in clients if client.width>=width_level]
            total_weight = sum(client.train_samples for client in eligible_clients)
            if total_weight == 0:
                continue

            # 调整聚合权重
            adjusted_weights = {}
            for client in eligible_clients:
                # find the cluster id of the client
                cluster_id = None
                for cid, clustered_clients in clusters.items():  # key: 簇id, value: 客户端列表
                    if client in clustered_clients:
                        cluster_id = cid
                        break

                if cluster_id is not None:
                    # clients with same cluster id would be assigned higher weight
                    delta_weight = 0.1  # 权重的增值
                    adjusted_weights[client.rank] = client.train_samples / total_weight * (1 + delta_weight)
                else:
                    adjusted_weights[client.rank] = client.train_samples / total_weight

            # normalize the adjusted weights
            total_adjusted_weight = sum(adjusted_weights.values())
            for client in eligible_clients:
                adjusted_weights[client.rank] /= total_adjusted_weight

            # 提取所有客户端当前层级的参数然后累加: 比如当前层级为0.5, 那么就提取模型[0.25, 0.5]之间的params
            for client in eligible_clients:
                # 第一个宽度层级直接提取该宽度下的参数
                if idx == 0:
                    delta_params = self.extract_width_params(client.params, width_level)
                # 后面的宽度层级需计算当前宽度与前一宽度的参数差异
                else:
                    prev_params = self.extract_width_params(client.params, sorted_widths[idx-1])
                    current_params = self.extract_width_params(client.params, width_level)
                    delta_params = {}  # 差异计算
                    for key in current_params:
                        # 如果当前参数在前一层级中存在, 且形状匹配
                        if key in prev_params and current_params[key].shape == prev_params[key].shape:
                            # 计算差异
                            delta_params[key] = current_params[key] - prev_params[key]
                        elif key in prev_params:
                            # 如果当前参数在前一层级中存在, 但形状不匹配, 只计算匹配的部分
                            min_shape = [min(a, b) for a, b in zip(current_params[key].shape, prev_params[key].shape)]
                            slice_indices = [slice(0, s) for s in min_shape]
                            delta_params[key] = current_params[key][tuple(slice_indices)] - prev_params[key][tuple(slice_indices)]
                        else:
                            # 如果当前参数在前一层级中不存在, 直接使用当前参数
                            delta_params[key] = current_params[key]

                # 当前层级参数提取出来之后进行加权累加
                weight = adjusted_weights[client.rank]
                for name in delta_params:
                    if name in layer_aggregates[width_level]:
                        # 如果维度匹配
                        if delta_params[name].shape == layer_aggregates[width_level][name].shape:
                            layer_aggregates[width_level][name] += delta_params[name].to(self.device)*weight
                        else:
                            # 如果维度不匹配, 只聚合匹配的部分
                            min_shape = [min(a, b) for a, b in zip(delta_params[name].shape, layer_aggregates[width_level][name].shape)]
                            slice_indices = [slice(0, s) for s in min_shape]
                            layer_aggregates[width_level][name][tuple(slice_indices)] += delta_params[name].to(self.device)[tuple(slice_indices)]*weight
            
            # 全局模型更新
            current_layer_state = global_model.state_dict()
            for name in current_layer_state:
                if name in layer_aggregates[width_level]:
                    current_layer_state[name] += layer_aggregates[width_level][name]
            global_model.load_state_dict(current_layer_state)

        # self.model.load_state_dict(global_model.state_dict())

        # 只加载形状匹配的参数到 self.model
        model_state = self.model.state_dict()
        global_state = global_model.state_dict()
        for key in model_state:
            if key in global_state and model_state[key].shape == global_state[key].shape:
                model_state[key] = global_state[key]
        self.model.load_state_dict(model_state)

        for client in self.selected_clients:
            client.last_params = copy.deepcopy(client.params)  # 保存当前参数供下一轮聚类使用
    """

    """
    # def multi_width_aggregate(self, clients=None):
    #     # 统一宽度场景的简化聚合
    #     avg_params = {}
    #     total_samples = sum(c.train_samples for c in clients)
        
    #     # 初始化累加器
    #     for name, param in self.model.named_parameters():
    #         avg_params[name] = torch.zeros_like(param.data)
        
    #     # 加权求和
    #     for client in clients:
    #         for name, param in client.params.items():
    #             if name in avg_params:
    #                 avg_params[name] += param.data * (client.train_samples / total_samples)
        
    #     # 更新全局模型
    #     self.model.load_state_dict(avg_params)
    """

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