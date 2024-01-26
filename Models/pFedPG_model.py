import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
import math
import random
import copy
import numpy as np
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from collections import Counter, OrderedDict

class client_prompted_vit_b32(nn.Module):
    def __init__(self, num_tokens, prompt_dropout_value=0.0, frozen_pretrian=True, num_classes=10):
        super(client_prompted_vit_b32, self).__init__()
        self.num_tokens = num_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout_value)
        self.prompt_proj = nn.Identity()
        self.vit_b32 = models.vit_b_32(weights='IMAGENET1K_V1')
        #self.classification_head = nn.Linear(self.vit_b32.heads.head.out_features, num_classes)
        self.trainable_keys = list()

        self.prompt_embeddings = nn.Parameter(torch.zeros(self.num_tokens,
                                                          self.vit_b32.encoder.layers[0].ln_1.weight.shape[0]))
        if frozen_pretrian:
            self.vit_b32.requires_grad_(False)

    def embedding_input(self, x):
        x = self.vit_b32._process_input(x)
        n = x.shape[0]

        batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x += self.vit_b32.encoder.pos_embedding
        x = self.vit_b32.encoder.dropout(x)
        return x

    def incorporate_prompt(self, x):
        batch_size = x.shape[0]
        x = self.embedding_input(x)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(torch.unsqueeze(self.prompt_embeddings, 0)).expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        self.trainable_keys = grad_keys

    def forward(self, x):
        embedding_output = self.incorporate_prompt(x)
        hidden_states = self.vit_b32.encoder.layers(embedding_output)
        encoded = self.vit_b32.encoder.ln(hidden_states)
        encoded = encoded[:, 0]
        encoded = self.vit_b32.heads(encoded)
        #logits = self.classification_head(encoded)
        return encoded
    
    
class classification_head(nn.Module):
    def __init__(self, n_input=1000, n_output=10, nonlinearity=False):
        super(classification_head, self).__init__()
        layers = []
        if nonlinearity:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_input, 512))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(512, n_output))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class BaseHeadsForLocal:
    def __init__(self, dataloaders, num_classes, local_lr, device='cpu'):
        self.dataloaders = dataloaders
        self.num_clients = len(dataloaders)
        self.num_classes = num_classes

        self.local_layers = [
            classification_head(n_output=num_classes).to(device) for _ in range(self.num_clients)
        ]
        self.local_optimizers = [
            optim.Adam(self.local_layers[i].parameters(),
                       local_lr, betas=(0.9, 0.98), eps=1e-6) for i in range(self.num_clients)
        ]

    def __len__(self):
        return self.num_clients
    
    
class prompt_generator(nn.Module):
    def __init__(self, num_tokens, num_clients, k_dim, v_dim, embed_dim=768, dropout_value=0.0):
        super(prompt_generator, self).__init__()
        self.scale = k_dim ** -0.5

        self.to_k = nn.Linear(embed_dim, k_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, v_dim, bias=False)
        self.to_q = nn.Linear(embed_dim, k_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(v_dim, embed_dim),
            nn.Dropout(dropout_value)
        )

        patch_size = _pair((32, 32))
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))
        self.base_prompts = nn.Parameter(torch.zeros(num_tokens, embed_dim))
        nn.init.uniform_(self.base_prompts.data, -val, val)
        #self.descriptor = nn.Parameter(torch.zeros(num_clients, embed_dim))
        self.descriptor = nn.Embedding(num_embeddings=num_clients, embedding_dim=embed_dim)
        nn.init.uniform_(self.descriptor.weight, -1, 1)

    def forward(self, x_id=torch.tensor([0], dtype=torch.long)):
        k = self.to_k(self.base_prompts) # Nt * lk
        v = self.to_v(self.base_prompts) # Nt * lv
        q = self.to_q(self.descriptor(x_id)) # 1 * lk

        dots = torch.matmul(q, k.T) * self.scale #1 * Nt
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v) #1 * lv
        Pn = self.base_prompts + self.to_out(out)
        return Pn