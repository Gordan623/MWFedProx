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
from utils.custom_models import create_model_instance
from FLamingo.core.utils.chores import merge_several_dicts
import torch.nn.functional as F
import copy


# class prox_loss(torch.nn.Module):
#     """
#     FedProx loss function
#     """
#     def __init__(self, mu):
#         """
#         :param mu: 正则化系数
#         """
#         super(prox_loss, self).__init__()
#         self.mu = mu
    
#     def forward(self, outputs, targets, local_model, global_model):
#         """
#         :param outputs: 模型输出
#         :param targets: 目标值
#         :param local_model: 当前客户端模型实例
#         :param global_model: 从服务器接收的全局模型实例
#         """
#         # 基本的交叉熵损失
#         cross_entropy_loss = F.cross_entropy(outputs, targets)
#         # FedProx 近似项
#         proximal_term = 0.0
#         # for w, w_star in zip(local_model_params, global_model_params):
#         #     proximal_term += (w - w_star).norm(2)

#         # ↑当客户端模型与全局模型宽度不同时，参数数量和形状不匹配，导致循环错位↑
#         # 按名称对齐参数，跳过 BN 层
#         local_params = dict(local_model.named_parameters())
#         global_params = dict(global_model.named_parameters())
#         for name, local_param in local_params.items():
#             # 跳过bn层
#             if 'bn' in name:
#                 continue
#             # 如果参数名称在全局模型中存在, 则计算正则项
#             if name in global_params:
#                 global_param = global_params[name]
#                 if local_param.shape == global_param.shape:
#                     # 计算L2范数
#                     proximal_term += (local_param - global_param).norm(2)
#                 else:
#                     self.log(f"Parameter shape mismatch for {name}: local {local_param.shape}, global {global_param.shape}")
#         # 计算正则化项
#         total_loss = cross_entropy_loss + (self.mu / 2.0) * proximal_term
#         return total_loss
    

class FedProxClient(Client):
    """
    FedProx Client.
    New defined:
        prox_loss: modified loss for FedProx, proximal term.
        capacity: save models not trained well.
    """

    def init(self):
        self.network = NetworkHandler()


    def train_iters(self, model, dataloader, loss_func, optimizer, scheduler=None, iters=None):
        """
        Train given dataset on given dataloader with given iters.
        Args:
            model: model to be trained
            dataloader: dataloader for the dataset
            loss_func: loss function
            optimizer: optimizer
            scheduler: default None, learning rate scheduler, lr will be consistent if not given
            iters: number of iterations
        Return:
            dict: train_loss, train_samples, train_time
        """
        model.train()
        epoch_loss, num_samples = 0.0, 0
        s_t = time.time()
        num_batches = 0
        inf_loader = infinite_dataloader(dataloader)
        if iters is None:
            iters = self.args.local_iteration

        for i in range(iters):
            data, target = next(inf_loader)
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()  
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()  
            optimizer.step() 
            batch_num_samples = len(target)
            epoch_loss += loss.item() * batch_num_samples  
            num_samples += batch_num_samples
            num_batches += 1
        if scheduler is not None:
            scheduler.step()  # 更新学习率
        
        if self.USE_SIM_SYSHET:
            train_time = num_batches * self.rand_comp()
        else:
            train_time = time.time() - s_t

        tcp = train_time

        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': train_time, 'tcp': tcp}
        # return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': train_time}


    def run(self):
        """
        Client jobs.
        """
        data = self.listen()
        self.width = data['width']       
        self.state_dict = data['state_dict']
        self.tau = 50
        self.model = create_model_instance(self.model_type, self.dataset_type, self.width).to(self.device)
        self.model.load_state_dict(self.state_dict)
        self.loss_func = torch.nn.CrossEntropyLoss() if self.dataset_type in ['mnist', 'cifar10'] else torch.nn.BCEWithLogitsLoss()
        self.dataset = ClientDataset(self.dataset_type, self.data_dir, self.rank)
        self.train_loader = self.dataset.get_train_loader(self.batch_size)
        self.test_loader = self.dataset.get_test_loader(self.test_batch_size)
        # self.test_loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)

        while True:
            data = self.listen()
            # print(data.keys())
            self.tau = data['tau']
            if data['status'] == 'STOP':
                if self.verb: self.log('Stopped by server')
                break
            elif data['status'] == 'TRAINING':
                self.model.load_state_dict(data['state_dict'])
                trained_info = self.train_iters(
                    self.model, self.train_loader, self.loss_func, self.optimizer, iters=self.tau
                    # self.model, self.train_loader, self.loss_func, self.optimizer, iters=self.local_iteration
                    )
                tested_info = self.test(
                    self.model, self.test_loader, self.loss_func, self.device)
                # Construct data to send
                data_to_send = merge_several_dicts([trained_info, tested_info])
                self.log(f"send data: {data_to_send}")
                data_to_send['state_dict'] = self.model.state_dict()
                self.send(data_to_send)
            else:
                raise Exception('Unknown status')
            
            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break

if __name__ == '__main__':
    client = FedProxClient()
    client.run()