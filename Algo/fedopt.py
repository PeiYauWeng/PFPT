import torch
from torch import nn
import copy
import numpy as np
from Algo.fedavg import fedavg
from Algo.communication import communication
from util.train_eval import train, evaluate
from util.print_info import print_epoch_end

class fedopt(fedavg):
    def __init__(self, server_model, scenario, loss_fun, global_lr, class_mask, fed_method='fedopt', nonpara_hidden=128, device='cuda'):
        super(fedopt, self).__init__(server_model, scenario, loss_fun, class_mask, fed_method, nonpara_hidden, device)
        self.server_model.train()
        self.global_optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.server_model.parameters()), global_lr,
                                                 betas=(0.9, 0.98), eps=1e-6)
        self.global_lr = global_lr
        self.device = device

    def server_aggre(self):
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()
        new_model = copy.deepcopy(self.server_model)
        new_model, self.client_model = communication(new_model, self.client_model,
                                                     self.selected_client_weights, self.fed_method,
                                                     nonpara_hidden=self.nonpara_hidden,
                                                     device=self.device)
        with torch.no_grad():
            for param, new_param in zip(self.server_model.parameters(), new_model.parameters()):
                if param.requires_grad == True:
                    param.grad = param.data - new_param.data
        self.server_model.train()
        self.global_optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.server_model.parameters()),
                                                 self.global_lr, betas=(0.9, 0.98), eps=1e-6)
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()
        del new_model