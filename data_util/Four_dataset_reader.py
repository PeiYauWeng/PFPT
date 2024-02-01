import torch
import torchvision
import torch.utils.data as Data
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
import os
import numpy as np
import random
import math
from collections import Counter, OrderedDict
from torchvision.datasets import VisionDataset, ImageFolder
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        return img.convert('RGB')
        
class MNISTM(VisionDataset):
    def __init__(self, root, subset_size, shift_label=0, train=True, transform=None, random_seed=0):
        self.shift_label = shift_label
        self.transform = transform
        self.train = train
        if self.train:
            data_file = "processed/mnist_m_train.pt"
        else:
            data_file = "processed/mnist_m_test.pt"
        self.classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        self.data, self.targets = torch.load(os.path.join(root, data_file))
        #self.targets = self.targets.tolist()
        np.random.seed(random_seed)
        index_list = list()
        for i in range(len(self.classes)):
            index_list.append(np.where(self.targets.numpy()==i)[0][:(subset_size//len(self.classes))])
        shuffle_index = np.concatenate(index_list, axis=0)
        np.random.shuffle(shuffle_index)
        self.targets = self.targets[shuffle_index].tolist()
        self.data = self.data[shuffle_index]
        
    def __getitem__(self, index):
        img, tgt = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, (tgt+self.shift_label)

    def __len__(self):
        return len(self.data)
        
class Fashion_RGB_reader(Dataset):
    def __init__(self, root, subset_size, shift_label=0, train=True, transform=None, random_seed=0):
        fashion_dataset = torchvision.datasets.FashionMNIST(root=root, train=train, download=False)
        self.shift_label = shift_label
        self.transform = transform
        self.classes = fashion_dataset.classes
        np.random.seed(random_seed)
        index_list = list()
        for i in range(len(self.classes)):
            index_list.append(np.where(np.array(fashion_dataset.targets)==i)[0][:(subset_size//len(self.classes))])
        shuffle_index = np.concatenate(index_list, axis=0)
        np.random.shuffle(shuffle_index)
        self.targets = fashion_dataset.targets[shuffle_index]
        self.dataset = Subset(fashion_dataset, shuffle_index)
        
    def __getitem__(self, index):
        sample, tgt = self.dataset[index]
        sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, (tgt+self.shift_label)
    
    def __len__(self):
        return len(self.dataset)
        
class Cinic10_reader(Dataset):
    def __init__(self, root, subset_size, shift_label=0, train=True, transform=None, random_seed=0):
        if train:
            cinic10_data = ImageFolder(root+'/train', loader=pil_loader)
        else:
            cinic10_data = ImageFolder(root+'/test', loader=pil_loader)
        self.shift_label = shift_label
        self.transform = transform
        self.classes = cinic10_data.classes
        np.random.seed(random_seed)
        index_list = list()
        for i in range(len(self.classes)):
            index_list.append(np.where(np.array(cinic10_data.targets)==i)[0][:(subset_size//len(self.classes))])
        shuffle_index = np.concatenate(index_list, axis=0)
        np.random.shuffle(shuffle_index)
        self.targets = np.array(cinic10_data.targets)[shuffle_index].tolist()
        self.dataset = Subset(cinic10_data, shuffle_index)
        
    def __getitem__(self, index):
        sample, tgt = self.dataset[index]
        #sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, (tgt+self.shift_label)
    
    def __len__(self):
        return len(self.dataset)
        
class MMAFEDB_reader(Dataset):
    def __init__(self, root, subset_size, shift_label=0, train=True, transform=None, random_seed=0):
        if train:
            mmafedb_data = ImageFolder(root+'/train', loader=pil_loader)
        else:
            mmafedb_data = ImageFolder(root+'/test', loader=pil_loader)
        self.shift_label = shift_label
        self.transform = transform
        self.classes = mmafedb_data.classes
        if subset_size%len(self.classes) != 0:
            subset_size = math.ceil(subset_size/len(self.classes))*len(self.classes)
        np.random.seed(random_seed)
        index_list = list()
        for i in range(len(self.classes)):
            index_list.append(np.where(np.array(mmafedb_data.targets)==i)[0][:(subset_size//len(self.classes))])
        shuffle_index = np.concatenate(index_list, axis=0)
        np.random.shuffle(shuffle_index)
        self.targets = np.array(mmafedb_data.targets)[shuffle_index].tolist()
        self.dataset = Subset(mmafedb_data, shuffle_index)
        
    def __getitem__(self, index):
        sample, tgt = self.dataset[index]
        #sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, (tgt+self.shift_label)
    
    def __len__(self):
        return len(self.dataset)
        
def four_dataset_reader(subset_size_list, train=True, transform=None):
    class_mask = list()
    mnistm = MNISTM(root='../Probabilistic_Prompt_Aggregation_ver2/Dataset/MNISTM', subset_size=subset_size_list[0], shift_label=0, train=train, transform=transform)
    fashsionRGB = Fashion_RGB_reader(root='../Probabilistic_Prompt_Aggregation_ver2/Dataset/FashionMNIST',subset_size=subset_size_list[1], 
                                     shift_label=10, train=train, transform=transform)
    cinic = Cinic10_reader(root='../Probabilistic_Prompt_Aggregation_ver2/Dataset/cinic10-py', subset_size=subset_size_list[2],
                           shift_label=20, train=train, transform=transform)
    mmafedb = MMAFEDB_reader('../Probabilistic_Prompt_Aggregation_ver2/Dataset/MMAFEDB', subset_size=subset_size_list[3], 
                             shift_label=30, train=train, transform=transform)
    dataset_list = [mnistm, fashsionRGB, cinic, mmafedb]
    nb_classes = 0
    for i in range(len(dataset_list)):
        class_mask.append([i + nb_classes for i in range(len(dataset_list[i].classes))])
        nb_classes += len(dataset_list[i].classes)
    return dataset_list, class_mask