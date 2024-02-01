import torch
from torch import nn
import copy
import numpy as np
from Algo.fedavg import fedavg
from util.train_eval import train, evaluate, train_prox
from util.print_info import print_epoch_end

class fedprox(fedavg):
    def __init__(self, server_model, scenario, loss_fun, mu, class_mask, fed_method='fedprox', nonpara_hidden=128, device='cuda'):
        super(fedprox, self).__init__(server_model, scenario, loss_fun, class_mask, fed_method, nonpara_hidden, device)
        self.mu = mu

    def client_train(self, comm_round, epochs, lr, output_file, opt_func=torch.optim.Adam, reduce_sim_scalar=0.01, print_output=False):
        if self.scenario.type == 'cross_devices':
            self.selected_client_index, self.selected_distributed_dataloaders, self.selected_client_weights \
            = self.scenario.cross_devices_random_selecting()
        for i in range(self.scenario.n_clients_each_round):
            torch.cuda.empty_cache()
            optimizer = opt_func(filter(lambda p : p.requires_grad, self.client_model[i].parameters()), lr,
                                 betas=(0.9, 0.98), eps=1e-6)
            if print_output:
                print(f'------------Client_{self.selected_client_index[i]+1} start local trainig------------',
                      file=output_file)
            if self.class_mask is not None:
                mask = self.class_mask[(self.selected_client_index[i]//20)]
                print(mask, file=output_file)
            for epoch in range(epochs):
                self.client_model[i].train()
                if comm_round > 0:
                    if self.class_mask is not None:
                        l, t, a = train_prox(self.client_model[i], self.server_model, self.selected_distributed_dataloaders[i], 
                                        optimizer, self.loss_fun, self.mu, self.device, reduce_sim_scalar, mask)
                    else:
                        l, t, a = train_prox(self.client_model[i], self.server_model, self.selected_distributed_dataloaders[i], 
                                        optimizer, self.loss_fun, self.mu, self.device, reduce_sim_scalar)
                else:
                    if self.class_mask is not None:
                        l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i], 
                                    optimizer, self.loss_fun, self.device, reduce_sim_scalar, mask)
                    else:
                        l, t, a = train(self.client_model[i], self.selected_distributed_dataloaders[i], 
                                        optimizer, self.loss_fun, self.device, reduce_sim_scalar)
                if print_output:
                    print_epoch_end(epoch, l, t, a, output_file)
            if hasattr(self.client_model[i], 'trained_prompts_checklist'):
                self.client_model[i].trained_prompts_checklist /= torch.max(self.client_model[i].trained_prompts_checklist)
                print(torch.nonzero(self.client_model[i].trained_prompts_checklist).flatten(), file=output_file)
                print(self.client_model[i].trained_prompts_checklist, file=output_file)