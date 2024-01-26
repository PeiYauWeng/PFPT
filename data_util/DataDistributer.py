import torch
import torch.utils.data as Data
import numpy as np
from collections import Counter, OrderedDict

class CustomSubset(Data.Subset):
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in self.indices]
        self.classes = len(np.unique(np.array(self.targets)))#dataset.classes
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.subset_transform:
            x = self.subset_transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

class DataPartitioner(object):
    def __init__(
        self, dataset, n_clients, seed=0
    ):
        self.dataset = dataset
        #self.preprocess = preprocess
        self.n_clients = n_clients
        #self.alpha =alpha
        self.seed = seed
        #self.least_samples = least_samples
        self.n_classes = len(self.dataset.classes)
        self.targets_numpy = np.array(self.dataset.targets, dtype=np.int32)
        self.class_idcs = [np.argwhere(self.targets_numpy == y).flatten()
                           for y in range(self.n_classes)]
        self.client_idcs = [[] for _ in range(self.n_clients)]
        '''if partition_type == 'dirichlet':
            self.dirichlet_split_noniid()
        else:
            self.distribution_iid()'''

    def dirichlet_split_noniid(self, alpha, least_samples, manual_seed=None):
        again = True
        if manual_seed is not None:
          self.seed = manual_seed
        while again:
            tmp_client_idcs = [[] for _ in range(self.n_clients)]
            #self.seed += 1
            np.random.seed(self.seed)
            label_distribution = np.random.dirichlet([alpha]*self.n_clients, self.n_classes)
            for k_idcs, fracs in zip(self.class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(k_idcs,
                                                  (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                                  astype(int))):
                    tmp_client_idcs[i] += [idcs]

            tmp_client_idcs = [np.concatenate(idcs) for idcs in tmp_client_idcs]
            check_again = np.array([len(tmp_client_idcs[i]) for i in range(self.n_clients)])
            if np.min(check_again) > least_samples:
                again = False
                self.client_idcs = tmp_client_idcs
            else:
                self.seed += 1
            del tmp_client_idcs

    def manual_allocating_noniid(self, n_dominated_classes_per_client, dominated_ratio, alpha):
        np.random.seed(self.seed)
        index = np.tile(np.arange(self.n_clients), n_dominated_classes_per_client).reshape(n_dominated_classes_per_client,
                                                                                           self.n_clients)
        for i in range(n_dominated_classes_per_client):
            np.random.shuffle(index[i])
        dominated_distribution = np.round(np.arange(n_dominated_classes_per_client*self.n_clients/self.n_classes,
                                                    n_dominated_classes_per_client*self.n_clients,
                                                    n_dominated_classes_per_client*self.n_clients/self.n_classes))

        dominated_distribution = np.split(index.flatten(), dominated_distribution[:(self.n_classes-1)].astype(int))

        for i in range(self.n_classes):
            dominated_part = self.class_idcs[i][:int(len(self.class_idcs[i])*dominated_ratio)]
            nondominated_part = self.class_idcs[i][int(len(self.class_idcs[i])*dominated_ratio):]
            for j, idcs in zip(dominated_distribution[i],
                               np.array_split(dominated_part, len(dominated_distribution[i]))):
                self.client_idcs[j] += [idcs]
            nondominated_part_distribution = np.random.dirichlet([alpha]*(self.n_clients-len(dominated_distribution[i])),
                                                                 1).flatten()
            nondominated_clients = np.setdiff1d(np.arange(self.n_clients), dominated_distribution[i], True)
            np.random.shuffle(nondominated_clients)
            for nj, n_idcs in zip(nondominated_clients,
                                  np.split(nondominated_part, (np.cumsum(nondominated_part_distribution)[:-1] \
                                                               *len(nondominated_part)).astype(int))):

                self.client_idcs[nj] += [n_idcs]
        self.client_idcs = [np.concatenate(idcs) for idcs in self.client_idcs]

    def distribution_iid(self):
        np.random.seed(self.seed)
        for k_idcs in self.class_idcs:
            evenly_distribution = np.round(np.arange(len(k_idcs)/self.n_clients, len(k_idcs), len(k_idcs)/self.n_clients))
            np.random.shuffle(k_idcs)
            for i , idcs in enumerate(np.split(k_idcs,
                                               evenly_distribution[:(self.n_clients-1)].astype(int))):
                self.client_idcs[i] += [idcs]
        self.client_idcs = [np.concatenate(idcs) for idcs in self.client_idcs]

    def get_distirbution_stats(self):
        stats = {}
        for i in range(self.n_clients):
            stats[i] = {"x": None, "y": None}
            stats[i]["x"] = len(self.targets_numpy[self.client_idcs[i]])
            stats[i]["y"] = Counter(self.targets_numpy[self.client_idcs[i]].tolist())

        '''num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
        stats["sample per client"] = {
                "std": num_samples.mean(),
                "stddev": num_samples.std(),
                }'''
        return stats

    def get_all_client_weights(self):
        all_client_weights = list()
        for i in range(self.n_clients):
            all_client_weights.append(len(self.targets_numpy[self.client_idcs[i]]))
        return np.array(all_client_weights)

    def get_distributed_data(self, batch_size, shuffle=True):
        distributed_dataloaders = list()
        for i in range(self.n_clients):
            #subset_dataset = CustomSubset(self.dataset, self.client_idcs[i], preprocess)
            subset_dataset = Data.Subset(self.dataset, torch.from_numpy(self.client_idcs[i]))
            distributed_dataloaders.append(Data.DataLoader(subset_dataset, batch_size=batch_size, 
                                                           shuffle=shuffle, num_workers=2))
        return distributed_dataloaders