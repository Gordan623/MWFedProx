# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os
import time

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)
# 应对显卡分配问题：CUDA out of memory
if rank < 28:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str((rank-28) % 2 + 2)

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')

# Now import FLamingo
from FLamingo.core.client import *
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from model_utils import create_model_instance
import torch.nn.functional as F
import copy


class prox_loss(torch.nn.Module):
    """
    FedProx loss function
    """
    def __init__(self, mu):
        """
        :param mu: 正则化系数
        """
        super(prox_loss, self).__init__()
        self.mu = mu
    
    def forward(self, outputs, targets, local_model, global_model):
        """
        :param outputs: 模型输出
        :param targets: 目标值
        :param local_model: 当前客户端模型实例
        :param global_model: 从服务器接收的全局模型实例
        """
        # 基本的交叉熵损失
        cross_entropy_loss = F.cross_entropy(outputs, targets)
        # FedProx 近似项
        proximal_term = 0.0
        # for w, w_star in zip(local_model_params, global_model_params):
        #     proximal_term += (w - w_star).norm(2)

        # ↑当客户端模型与全局模型宽度不同时，参数数量和形状不匹配，导致循环错位↑
        # 按名称对齐参数，跳过 BN 层
        local_params = dict(local_model.named_parameters())
        global_params = dict(global_model.named_parameters())
        for name in local_params:
            if name in global_params and 'bn' not in name:  # 跳过 BN 层
                w = local_params[name]
                w_star = global_params[name]
                proximal_term += torch.norm(w - w_star, p=2)
        total_loss = cross_entropy_loss + (self.mu / 2.0) * proximal_term
        return total_loss
    

class FedProxClient(Client):
    """
    FedProx Client.
    New defined:
        prox_loss: modified loss for FedProx, proximal term.
        capacity: save models not trained well.
    """

    def init(self):
        """
        self.width = 1.0
        self.model = create_model_instance(self.model_type, self.dataset_type, self.width)
        self.model = self.model.to(self.device)
        self.loss_func = prox_loss(self.mu)
        self.test_loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        """
        
        self.width = None
        self.model = None
        self.loss_func = None
        self.test_loss_func = None
        self.optimizer = None
        self.lr_scheduler = None
        self.params = None
        self.train_samples = None
        self.local_iteration = None
        

    def train(self):
        """
        Use costumized loss func to train.
        """
        model, train_loader = self.model, self.train_loader
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss, num_samples = 0.0, 0
        train_start_time = time.time()  # 记录训练开始时间
        inf_loader = infinite_dataloader(train_loader)
        # 按迭代次数训练
        for iter in range(self.local_iteration):
            data, target = next(inf_loader)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  
            output = model(data)

            loss = self.loss_func(output, target, model, global_model)

            loss.backward()  
            self.optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # 更新学习率
        train_end_time = time.time()  # 记录训练结束时间
        train_time = train_end_time - train_start_time  # 计算训练时间
        self.train_samples = num_samples
        return {
            'train_loss': epoch_loss / num_samples,
            'train_samples': num_samples,
            'train_time': train_time
        }

    
    def set_parameter(self, params_dict, model = None):
        """
        Set model parameters from a dictionary of parameters.
        Args:
            params_dict: Dictionary containing parameter names and values.
            model: Model to set parameters for. Defaults to self.model.
        """
        model = self.model if model is None else model
        model.load_state_dict(params_dict)

    def run(self):
        """
        Client jobs.
        """
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                
                # 判断是否为首次初始化
                if self.width != data['width'] or self.model is None:
                    # print(f"before init, for rank {rank}, self.width = {self.width}, data['width'] = {data['width']}")
                    # 首次初始化需要init_model
                    self.width = data['width']
                    # print(f"after init, for rank {rank}, self.width = {self.width}, data['width'] = {data['width']}")
                    
                    # ↑宽度已正确传输↑
                    self.local_iteration = data['local_iteration']
                    
                    # TODO: change model width
                    del self.model, self.optimizer, self.lr_scheduler
                    # torch.cuda.empty_cache()
                    # data = self.listen()
                    # self.width = data['width']
                    # TODO: re-init model
                    self.model = create_model_instance(self.model_type, self.dataset_type, self.width)
                    self.model = self.model.to(self.device)
                    self.loss_func = prox_loss(self.mu)
                    self.test_loss_func = torch.nn.CrossEntropyLoss()
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
                    self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)          

                # 加载服务器下发的参数（允许部分匹配）
                missing, unexpected = self.model.load_state_dict(data['params'], strict=False)
                self.log(f"Loaded params: missing={missing}, unexpected={unexpected}")

                # self.print_model_info()
                # TODO:  没啥必要
                self.set_parameter(data['params'], self.model)
                self.status = data['status']
                self.local_epochs = data['local_epochs']
                self.straggler = data['straggler']
                self.log(f"I'm {'straggler' if self.straggler else 'not straggler'}, training with width {self.width}, local epochs {self.local_epochs}")
                bf_test_dic = self.test(
                    self.model, self.test_loader, self.test_loss_func, self.device)
                train_dic = self.train()
                # self.log("Training finish.")
                af_test_dic = self.test(
                    self.model, self.test_loader, self.test_loss_func, self.device)
                data_to_send = merge_several_dicts(
                    [af_test_dic, train_dic]
                )
                data_to_send.update({
                    'bf_acc': bf_test_dic['test_acc'],
                    'width': self.width,
                    'bf_loss': bf_test_dic['test_loss'],
                    'params': self.model.state_dict()
                })
                # -----2025.03.20修改-----
                self.params = data_to_send['params']
                # -----------------------
                self.send(data_to_send)
                # self.log("Send success")
            elif data['status'] == 'STOP':
                print('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
        # out of the loop
        print('stopped')


if __name__ == '__main__':
    client = FedProxClient()
    client.run()