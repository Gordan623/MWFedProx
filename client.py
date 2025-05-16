# All os.environ['CUDA_VISIBLE_DEVICES'] are set at the beginning
from mpi4py import MPI
import os
import time

WORLD = MPI.COMM_WORLD
rank = WORLD.Get_rank()
size = WORLD.Get_size()

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

class FedProxClient(Client):
    """
    FedProx Client.
    New defined:
        prox_loss: modified loss for FedProx, proximal term.
        capacity: save models not trained well.
    """

    def init(self):
        self.network = NetworkHandler()
        # self.log(self.rank)
        self.syshet = self.sys_het_list[(self.rank - 1) % len(self.sys_het_list)]
        # self.log(self.syshet)
        self.tcp = self.syshet['computation'] / 10
        self.tcm = self.syshet['communication']
        self.cp_dynamic = self.syshet['cp_dynamics']
        self.cm_dynamic = self.syshet['cm_dynamics']

    def rand_comp(self):
        # 随机从 [self.tcp * (1-self.cp_dynamic/10) 到 self.tcp * (1+self.cp_dynamic/10)] 中间采样一个数
        random_time = random.uniform(
            self.tcp * (1 - self.cp_dynamic / 10),
            self.tcp * (1 + self.cp_dynamic / 10)
        )
        return random_time
    
    def rand_send(self):
        # 随机从 [self.tcm * (1-self.cm_dynamic/10) 到 self.tcm * (1+self.cm_dynamic/10)] 中间采样一个数
        random_time = random.uniform(
            self.tcm * (1 - self.cm_dynamic / 10),
            self.tcm * (1 + self.cm_dynamic / 10)
        )
        return random_time

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
            iters = self.args.local_iters

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

        return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': train_time}
        # return {'train_loss': epoch_loss/num_samples, 'train_samples': num_samples, 'train_time': train_time}


    def run(self):
        """
        Client jobs.
        """
        # data = self.listen()
        # self.width = data['width']       
        # self.state_dict = data['state_dict']
        # self.tau = 50
        # self.model = create_model_instance(self.model_type, self.dataset_type, self.width).to(self.device)
        # self.model.load_state_dict(self.state_dict)
        self.loss_func = torch.nn.CrossEntropyLoss() if self.dataset_type in ['mnist', 'cifar10'] else torch.nn.BCEWithLogitsLoss()
        self.dataset = ClientDataset(self.dataset_type, self.data_dir, self.rank)
        self.train_loader = self.dataset.get_train_loader(self.batch_size)
        self.test_loader = self.dataset.get_test_loader(self.test_batch_size)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)

        while True:
            data = self.listen()
            if data['status'] == 'STOP':
                if self.verb: self.log('Stopped by server')
                break
            elif data['status'] == 'TRAINING':
                self.width = data['width']
                self.local_iters = data['local_iters']
                # self.model.load_state_dict(data['state_dict'])  # 直接load会出现shape dismatch, 因为传入了新的width

                # 1. 先根据新传入的width重构模型
                self.model = create_model_instance(self.model_type, self.dataset_type, self.width).to(self.device)
                self.model.load_state_dict(data['state_dict'])

                # # 2. 重构optimizer&scheduler保证参数指向新的model
                # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
                # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)

                # 测试新模型结构
                #self.print_model_info(self.model)
                self.log(f"Current model width: {self.width}")
                if self.momentum != -1:
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=False)
                else:
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(), lr=self.lr, nesterov=False)
                # test_before_train_info = self.test(
                #     self.model, self.test_loader, self.loss_func, self.device)
                trained_info = self.train_iters(
                    self.model, self.train_loader, self.loss_func, self.optimizer, iters=self.local_iters)
                tested_info = self.test(
                    self.model, self.test_loader, self.loss_func, self.device)
                # Construct data to send
                data_to_send = merge_several_dicts([trained_info, tested_info])
                # data_to_send = {
                #     'acc_before_train': test_before_train_info['test_acc'],
                #     'loss_before_train': test_before_train_info['test_loss'],
                #     'test_acc': tested_info['test_acc'],
                #     'test_loss': tested_info['test_loss'],
                #     'train_loss': trained_info['train_loss'],
                #     'train_samples': trained_info['train_samples'],
                #     'train_time': trained_info['train_time'],
                #     'tcp': trained_info['tcp'],
                #     'test_samples': tested_info['test_samples'],
                # }
                if self.USE_SIM_SYSHET:
                    # send time usually larger than computation time
                    data_to_send['send_time'] = self.rand_send()
                self.log(f"send data: {data_to_send}")
                data_to_send['state_dict'] = self.model.state_dict()
                self.send(data_to_send)
                self.lr = max(self.lr * self.lr_decay, 5e-3)
            else:
                raise Exception('Unknown status')


            self.finalize_round()
            if self.global_round >= self.max_epochs:
                if self.verb: self.log(f'Reaching epochs limit {self.max_epochs}')
                break
            
        if self.verb: self.log(f'finished at round {self.global_round}')

if __name__ == '__main__':
    client = FedProxClient()
    client.run()