from torchvision.datasets import VisionDataset, ImageFolder
from PIL import Image
import os
import math
import torch

def pil_loader(path):
    with open(path, 'rb') as f:
        img=Image.open(f)
        return img.convert('RGB')
        
class Pacs_reader(VisionDataset):
    def __init__(self, root, split='', train=True, transform=None, random_seed=0):
        self.split = split
        self.root = os.path.join(root,self.split)
        self.transform = transform
        
        allset = ImageFolder(self.root, transform=transform, loader=pil_loader)
        #print(allset.targets)
        train_len = math.ceil(len(allset)*0.8)
        test_len = len(allset) - train_len
        trainset, testset = torch.utils.data.random_split(allset, 
                                                          [train_len, test_len],
                                                          generator=torch.Generator().manual_seed(random_seed))
        if train:
            self.dataset = trainset
        else:
            self.dataset = testset
        
        if split == 'art_painting':
            self.appended_label = 0
        elif split == 'cartoon':
            self.appended_label = 7
        elif split == 'photo':
            self.appended_label = 14
        else:
            self.appended_label = 21
            
        self.targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        self.classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __getitem__(self, index):
        sample, tgt = self.dataset[index]
        return sample, (tgt+self.appended_label)

    def __len__(self):
        return len(self.dataset)