import torch
from torch import nn
import copy
import numpy as np
from collections import OrderedDict
from Algo.fedavg import fedavg
from Algo.communication import communication
from util.train_eval import train, evaluate, train_scaffold
from util.print_info import print_epoch_end

class scaffold(fedavg):
    def __init__(self, server_model, scenario, loss_fun, class_mask, fed_method='scaffold', nonpara_hidden=128, device='cuda'):
        super(scaffold, self).__init__(server_model, scenario, loss_fun, class_mask, fed_method, nonpara_hidden, device)
        #self.device = device
        self.personalized_model_weights = self.scenario.init_personalized_model_weights(server_model, device=device)
        self.personalized_control = [OrderedDict() for _ in range(self.scenario.n_clients)]
        self.personalized_delta_control = [OrderedDict() for _ in range(self.scenario.n_clients)]
        self.personalized_delta_y = [OrderedDict() for _ in range(self.scenario.n_clients)]
        self.device = device
        
    #def init_contorl_parameters_storage(self, device='cuda'):
        for i in range(self.scenario.n_clients):
            for key in self.server_model.trainable_keys:
                self.personalized_control[i][key] = torch.zeros_like(self.server_model.state_dict()[key],
                                                                     dtype=torch.float32)#.to(device)
                self.personalized_delta_control[i][key] = torch.zeros_like(self.server_model.state_dict()[key],
                                                                     dtype=torch.float32)#.to(device)
                self.personalized_delta_y[i][key] = torch.zeros_like(self.server_model.state_dict()[key],
                                                                     dtype=torch.float32)#.to(device)

    def reconnect2current_models(self):
        for i, s_id in enumerate(self.selected_client_index):
            with torch.no_grad():
                for key in self.server_model.trainable_keys:
                    self.client_model[i].state_dict()[key].data.copy_(self.personalized_model_weights[s_id][key].data)
                    self.client_model[i].control[key].data.copy_(self.personalized_control[s_id][key].data)
                    self.client_model[i].delta_control[key].data.copy_(self.personalized_delta_control[s_id][key].data)
                    self.client_model[i].delta_y[key].data.copy_(self.personalized_delta_y[s_id][key].data)

    def reconnect2personalized_model_weights(self):
        for i, s_id in enumerate(self.selected_client_index):
            with torch.no_grad():
                for key in self.server_model.trainable_keys:
                    self.personalized_model_weights[s_id][key].data.copy_(self.client_model[i].state_dict()[key].data)
                    self.personalized_control[s_id][key].data.copy_(self.client_model[i].control[key].data)
                    self.personalized_delta_control[s_id][key].data.copy_(self.client_model[i].delta_control[key].data)
                    self.personalized_delta_y[s_id][key].data.copy_(self.client_model[i].delta_y[key].data)

    def update_client_controls(self, local_epochs, current_lr):
        #temp = OrderedDict({key: None for key in self.server_model.trainable_keys})
        #for key in self.server_model.trainable_keys:
            #temp[key] = copy.deepcopy(model.state_dict()[key].data)
        for i, s_id in enumerate(self.selected_client_index):
            self.client_model[i].eval()
            with torch.no_grad():
                for key in self.server_model.trainable_keys:
                    local_steps = local_epochs * len(self.selected_distributed_dataloaders[i])
                    self.client_model[i].control[key] = self.client_model[i].control[key] - self.server_model.control[key] \
                                 + (self.personalized_model_weights[s_id][key].data - self.client_model[i].state_dict()[key].data)/(local_steps*current_lr)
                    self.client_model[i].delta_y[key] = self.client_model[i].state_dict()[key].data - self.personalized_model_weights[s_id][key].data
                    self.client_model[i].delta_control[key] = self.client_model[i].control[key] - self.personalized_control[s_id][key].data

        #del before_train_model_dict, before_train_model_control, temp
    def client_train(self, comm_round, epochs, lr, output_file, opt_func=torch.optim.SGD, print_output=False):
        if self.scenario.type == 'cross_devices':
            self.selected_client_index, self.selected_distributed_dataloaders, self.selected_client_weights \
            = self.scenario.cross_devices_random_selecting()
        self.reconnect2current_models()
        for i in range(self.scenario.n_clients_each_round):
            torch.cuda.empty_cache()
            optimizer = opt_func(filter(lambda p : p.requires_grad, self.client_model[i].parameters()), lr)
                                 #betas=(0.9, 0.98), eps=1e-6)
            if print_output:
                print(f'------------Client_{self.selected_client_index[i]+1} start local trainig------------',
                      file=output_file)
            if self.class_mask is not None:
                mask = self.class_mask[(self.selected_client_index[i]//20)]
                print(mask, file=output_file)
            for epoch in range(epochs):
                self.client_model[i].train()
                if self.class_mask is not None:
                    l, t, a = train_scaffold(self.client_model[i], self.server_model,
                                             self.selected_distributed_dataloaders[i], 
                                             optimizer, self.loss_fun, epochs, self.device, mask)
                else:
                    l, t, a = train_scaffold(self.client_model[i], self.server_model,
                                             self.selected_distributed_dataloaders[i], 
                                             optimizer, self.loss_fun, epochs, self.device)
                #l, t, a = train_scaffold(self.client_model[i], self.server_model,
                                         #self.selected_distributed_dataloaders[i],
                                         #optimizer, self.loss_fun, epochs, self.device)
                if print_output:
                    print_epoch_end(epoch, l, t, a, output_file)
        self.update_client_controls(epochs, l)
        #self.reconnect2personalized_model_weights()

    def server_aggre(self):
        self.server_model, self.client_model = communication(self.server_model, self.client_model,
                                                             self.selected_client_weights, self.fed_method, 
                                                             self.scenario.n_clients, self.device)
        #print(f'Store aggre model to personalized stoarage')
        self.reconnect2personalized_model_weights()