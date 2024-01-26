import torch
from torch import nn
import copy
from communication import communication
from util.train_eval import train, evaluate
from util.print_info import print_epoch_end

class isolated_training(nn.Module):
    def __init__(self, server_model, scenario, loss_fun):
        super(isolated_training, self).__init__()
        self.server_model = server_model
        self.server_model.eval()
        self.scenario = scenario
        self.loss_fun = loss_fun
        self.history = list()

    def client_train_eval(self, epochs, lr, opt_func=torch.optim.Adam, print_output=False):
        for i in range(self.scenario.n_clients):
            torch.cuda.empty_cache()
            client_model = copy.deepcopy(server_model).to('cuda')
            optimizer = opt_func(filter(lambda p : p.requires_grad, client_model.parameters()), lr,
                                 betas=(0.9, 0.98), eps=1e-6)
            if print_output:
                print(f'------------Client_{i+1} start local trainig------------')
            for epoch in range(epochs):
                client_model.train()
                l, t, a = train(client_model, self.scenario.distributed_dataloaders[i], optimizer, self.loss_fun)
                if print_output:
                    print_epoch_end(epoch, l, t, a)
            test_loss, test_acc = evaluate(client_model, testloader, self.loss_fun)
            print(f'Client_{i+1}: Train_loss: {test_loss}; Accuracy: {test_acc}')
            self.history.append(test_acc)
            del client_model

    def show_top_10_test_accuracy(self):
        top_10 = sorted(self.history, reverse=True)[:10]
        print(f'top_10_test_accuracy: {np.mean(np.array(top_10))}')

    def show_all_test_average_accuracy(self):
        print(f'average_test_accuracy: {np.mean(np.array(self.history))}')