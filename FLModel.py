# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

import numpy as np
import copy
import os


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, T, batch_size, q, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.noise = None
        self.lr = lr
        self.E = E
        self.T = T
        self.clip = clip
        self.q = q
        self.model = model(data[0].shape[1], output_size).to(self.device)
        self.sigma = sigma

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            while len(idx) < 1:
                idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            optimizer.zero_grad()
            
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad
                    self.model.zero_grad()
                    
            # add Gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma, device=self.device)
            
            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size*self.q)
                
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            optimizer.step()
            
    def update_nondp(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            while len(idx) < 1:
                idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            optimizer.zero_grad()
            
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                loss.backward()
  
            optimizer.step()
            self.model.zero_grad()


class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']  # (float) C in [0, 1]
        self.epsilon = fl_param['eps']  # (ε, δ)-DP
        self.delta = fl_param['delta']
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations
        self.pth = fl_param['pth']  # model save path

        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']
        
        self.sigma = fl_param['noise']
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['tot_T'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 self.sigma,
                                 self.device)
                        for i in range(self.client_num)]
        self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())
        
        self.pth += self.global_model.__class__.__name__ + "/"
        if not os.path.exists(self.pth):
            os.makedirs(self.pth)
        
        pth = self.pth + "init/"
        if not os.path.exists(pth):
            os.makedirs(pth)
            
        pth = pth + "eps" + str(self.epsilon)+'/'
        if not os.path.exists(pth):
            os.makedirs(pth)
        torch.save(self.global_model.state_dict(), pth+"init.pth")

    def aggregated(self, idxs_users, e):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        
        # save model
        pth = self.pth + "epoch" + str(e) + "/"
        if not os.path.exists(pth):
            os.makedirs(pth)
            
        pth = pth + "eps" + str(self.epsilon)+'/'
        if not os.path.exists(pth):
            os.makedirs(pth)
            
        for i, local_m in enumerate(model_par):
            torch.save(local_m, pth+str(i)+".pth")
            
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        
        torch.save(self.global_model.state_dict(), pth+"global.pth")
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        # compute accuracy using test set
        self.global_model.eval()
        t_pred_y = self.global_model(self.data)
        _, predicted = torch.max(t_pred_y, 1)

        acc = (predicted == self.target).sum().item() / self.target.size(0)
        return acc

    def test_acc_femnist(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i].float())
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self, e):
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        #print(idxs_users)
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users, e))
        # acc = self.test_acc()
        acc = self.test_acc_femnist()
        return acc


    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr
