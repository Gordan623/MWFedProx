# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os
import time

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4)
# 应对显卡分配问题：CUDA out of memory
# if rank < 28:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4 + 4)
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str((rank-28) % 2 + 2 + 4)
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 4 + 4)

import sys
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('/data/yxli/FLamingo/')

# Now import FLamingo
from FLamingo.core.client import *
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from FLamingo.core.utils.data_utils import ClientDataset
from utils.model_utils import create_model_instance
from FLamingo.core.utils.chores import merge_several_dicts
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
        for name, local_param in local_params.items():
            # 跳过bn层
            if 'bn' in name:
                continue
            # 如果参数名称在全局模型中存在, 则计算正则项
            if name in global_params:
                global_param = global_params[name]
                if local_param.shape == global_param.shape:
                    # 计算L2范数
                    proximal_term += (local_param - global_param).norm(2)
                else:
                    self.log(f"Parameter shape mismatch for {name}: local {local_param.shape}, global {global_param.shape}")
        # 计算正则化项
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
        self.network = NetworkHandler()
        self.dataset = ClientDataset(self.dataset_type, self.data_dir, self.rank)
        self.train_loader = self.dataset.get_train_loader(self.batch_size)
        self.test_loader = self.dataset.get_test_loader(self.test_batch_size)
        

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
        data = self.listen()
        self.width = data['width']       
        self.local_iteration = data['local_iteration']
        self.state_dict = data['state_dict']
        del self.model, self.optimizer, self.lr_scheduler
        self.model = create_model_instance(self.model_type, self.dataset_type, self.width).to(self.device)
        self.loss_func = prox_loss(self.mu)
        self.test_loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)

        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                print(f"Client stopping...")
                global_round = data['global_round']      
                # 前两轮打印模型信息
                if global_round < 2:
                    self.log(f"[Round {global_round}] Client {self.rank} model info:")
                    self.print_model_info()
                self.model.load_state_dict(data['state_dict'])
                self.status = data['status']
                self.local_epochs = data['local_epochs']
                self.straggler = data['straggler']
                self.log(f"I'm {'straggler' if self.straggler else 'not straggler'}, training with width {self.width}, local epochs {self.local_epochs}")


                print(f"Client {self.rank} starts training & testing...")
                trained_info = self.train_iters(
                    self.model, self.train_loader, self.loss_func, self.optimizer, iters=self.local_iters)
                tested_info = self.test(
                    self.model, self.test_loader, self.loss_func, self.device)   
                print(f"Client {self.rank} finished training & testing.")            

                data_to_send = merge_several_dicts([trained_info, tested_info])
                # self.log(f"send data: {data_to_send}")
                # data_to_send['state_dict'] = self.model.state_dict()

                # bf_test_dic = self.test(
                #     self.model, self.test_loader, self.test_loss_func, self.device)
                # train_dic = self.train()

                # af_test_dic = self.test(
                #     self.model, self.test_loader, self.test_loss_func, self.device)
                # data_to_send = merge_several_dicts(
                #     [af_test_dic, train_dic]
                # )
                # data_to_send.update({
                #     'bf_acc': bf_test_dic['test_acc'],
                #     'width': self.width,
                #     'bf_loss': bf_test_dic['test_loss'],
                #     'params': self.model.state_dict()
                # })
                self.params = data_to_send['params']
                self.send(data_to_send)
            elif data['status'] == 'STOP':
                print('stop training...')
                break
            
            # finish the round as you wish
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        # out of the loop
        if self.verb: self.log(f'finished at round {self.global_round}')
        print('stopped')


if __name__ == '__main__':
    client = FedProxClient()
    client.run()