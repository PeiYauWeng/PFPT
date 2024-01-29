import torch
from torch import nn
import torchvision.models as models
from collections import Counter, OrderedDict

class vit_32_specific_classes(nn.Module):
    def __init__(self, classification_adaptor=True, freezing_pretrain=False, num_classes=10):
        super(vit_32_specific_classes, self).__init__()
        self.trainable_keys = list()
        self.control = OrderedDict()
        self.delta_control = OrderedDict()
        self.delta_y = OrderedDict()
        self.vit_b32 = models.vit_b_32(weights='IMAGENET1K_V1')
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.vit_b32.heads.head.out_features, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.vit_b32.heads.head.out_features, num_classes)
        if freezing_pretrain:
            self.vit_b32.requires_grad_(requires_grad=False)
            
    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        self.trainable_keys = grad_keys

    def init_contorl_parameter_for_scaffold(self, device='cuda'):
        if len(self.trainable_keys) == 0:
            raise ValueError("Forget initializing trainable keys list")
        for key in self.trainable_keys:
            self.control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_y[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)

    def forward(self, x):
        x = self.vit_b32(x)
        x = self.classification_head(x)
        return x
    
    
class resnet18_specific_classes(nn.Module):
    def __init__(self, freezing_pretrain=True, num_classes=10):
        super(resnet18_specific_classes, self).__init__()
        self.ResNet_model = models.resnet18(weights='IMAGENET1K_V1')
        self.control = OrderedDict()
        self.delta_control = OrderedDict()
        self.delta_y = OrderedDict()
        #self.ResNet_model.requires_grad_(requires_grad=False)
        specific_classes_fc = nn.Sequential(
            nn.Linear(self.ResNet_model.fc.in_features, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        if freezing_pretrain:
            self.ResNet_model.requires_grad_(requires_grad=False)
            self.ResNet_model.fc = specific_classes_fc
            self.trainable_keys = ['ResNet_model.fc.'+key for key in specific_classes_fc.state_dict().keys()]
        else:
            self.ResNet_model.fc = specific_classes_fc
            self.trainable_keys = ['ResNet_model.'+key for key in self.ResNet_model.state_dict().keys()] \
                                  +['ResNet_model.fc.'+key for key in specific_classes_fc.state_dict().keys()]
        #self.ResNet_model.fc = nn.Linear(self.ResNet_model.fc.in_features, num_classes)

    def init_contorl_parameter_for_scaffold(self, device='cuda'):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        self.trainable_keys = grad_keys
        for key in self.trainable_keys:
            self.control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_y[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)

    def forward(self, x):
        x = self.ResNet_model(x)
        return x
