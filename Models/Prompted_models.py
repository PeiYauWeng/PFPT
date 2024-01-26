import torch
from torch import nn
import torchvision.models as models
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from collections import Counter, OrderedDict

class Prompted_ViT_B32(nn.Module):
    def __init__(self, weight_init, prompt_method, num_tokens, prompt_dropout_value=0.0,
                 classification_adaptor=True, frozen_pretrian=True, num_classes=10):
        super(Prompted_ViT_B32, self).__init__()
        self.weight_init = weight_init
        self.prompt_method = prompt_method
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        patch_size = _pair((32, 32))
        self.prompt_dropout = nn.Dropout(prompt_dropout_value)
        self.prompt_proj = nn.Identity()
        self.vit_b32 = models.vit_b_32(weights='IMAGENET1K_V1')
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.vit_b32.heads.head.out_features, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.vit_b32.heads.head.out_features, num_classes)
        hidden_size_each_layers = self.record_hidden_size_each_layers(self.vit_b32)
        self.trainable_keys = list()
        self.control = OrderedDict()
        self.delta_control = OrderedDict()
        self.delta_y = OrderedDict()

        if weight_init == 'random':
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + hidden_size_each_layers[0]))
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size_each_layers[0]))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_method == 'deep':
                self.deep_prompt_embeddings_list = nn.ParameterList()
                for i in range(1, len(hidden_size_each_layers)):
                    val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + hidden_size_each_layers[i]))
                    deep_prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size_each_layers[i]))
                    nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
                    self.deep_prompt_embeddings_list.append(deep_prompt_embeddings)

        else:
            raise ValueError("Initiation is not supported")

        if frozen_pretrian:
            self.vit_b32.requires_grad_(False)

    def record_hidden_size_each_layers(self, origin_model):
        num_encoder_layers = len(origin_model.encoder.layers)
        hidden_size_record = list()
        for i in range(num_encoder_layers):
            for n, p in origin_model.encoder.layers[i].named_parameters():
                if 'ln_1.weight' in n:
                    hidden_size_record.append(p.shape[0])
        return hidden_size_record

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
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def forward_deep_prompt(self, embedding_output):
        batch_size = embedding_output.shape[0]
        for i in range(len(self.vit_b32.encoder.layers)):
            if i == 0:
                hidden_states = self.vit_b32.encoder.layers[i](embedding_output)
            else:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                self.deep_prompt_embeddings_list[i-1]).expand(batch_size, -1, -1))

                hidden_statese = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompt_emb,
                    hidden_states[:, (1+self.num_tokens):, :]
                ), dim=1)

                hidden_states = self.vit_b32.encoder.layers[i](hidden_states)
        encoded = self.vit_b32.encoder.ln(hidden_states)
        return encoded

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
        embedding_output = self.incorporate_prompt(x)
        if self.prompt_method == 'deep':
            encoded = self.forward_deep_prompt(embedding_output)
        elif self.prompt_method == 'shallow':
            hidden_states = self.vit_b32.encoder.layers(embedding_output)
            encoded = self.vit_b32.encoder.ln(hidden_states)

        encoded = encoded[:, 0]
        encoded = self.vit_b32.heads(encoded)
        logits = self.classification_head(encoded)
        return logits
    
    
class Prompted_ResNet18(nn.Module):
    def __init__(self, weight_init, num_tokens, cropsize, normalization_stats,
                 location='below', freezing_pretrain=True, num_classes=10):
        super(Prompted_ResNet18, self).__init__()
        self.weight_init = weight_init
        self.num_tokens = num_tokens
        self.cropsize = cropsize
        self.normalization_stats = normalization_stats
        self.location = location
        self.trainable_keys = list()
        self.control = OrderedDict()
        self.delta_control = OrderedDict()
        self.delta_y = OrderedDict()
        origin_model = models.resnet18(weights='IMAGENET1K_V1')
        if self.location == 'pad':
            self.setup_prompt_pad()
            self.prompt_layers = nn.Identity()
            self.frozen_layers = nn.Sequential(OrderedDict([
                ('conv1', origin_model.conv1),
                ('bn1', origin_model.bn1),
                ("relu", origin_model.relu),
                ("maxpool", origin_model.maxpool),
                ("layer1", origin_model.layer1),
                ("layer2", origin_model.layer2),
                ("layer3", origin_model.layer3),
                ("layer4", origin_model.layer4),
                ("avgpool", origin_model.avgpool),
        ]))
        elif self.location == 'below':
            origin_model = self.setup_prompt_below(origin_model)
            self.prompt_layers = nn.Sequential(OrderedDict([
                ("conv1", origin_model.conv1),
                ("bn1", origin_model.bn1),
                ("relu", origin_model.relu),
                ("maxpool", origin_model.maxpool),
            ]))
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("layer1", origin_model.layer1),
                ("layer2", origin_model.layer2),
                ("layer3", origin_model.layer3),
                ("layer4", origin_model.layer4),
                ("avgpool", origin_model.avgpool),
            ]))
        else:
            raise ValueError("Not supported")
        if freezing_pretrain:
            self.frozen_layers.requires_grad_(requires_grad=False)
        self.tuned_layers = nn.Identity()
        #self.prompt_layers = nn.Identity()
        self.specific_classes_fc = nn.Sequential(
            nn.Linear(origin_model.fc.in_features, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def setup_prompt_pad(self):
        if self.weight_init == "random":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.cropsize + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.cropsize, 2 * self.num_tokens
            ))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = transforms.Normalize(*self.normalization_stats)

        elif self.weight_init == "gaussian":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.cropsize + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.cropsize, 2 * self.num_tokens
            ))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Not supported")
        #return prompt_embeddings_tb, prompt_embeddings_lr, prompt_norm

    def setup_prompt_below(self, model):
        if self.weight_init == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cropsize, self.cropsize
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = transforms.Normalize(
                mean=[sum(self.normalization_stats[0])/3] * self.num_tokens,
                std=[sum(self.normalization_stats[1])/3] * self.num_tokens,
            )

        elif self.weight_init == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cropsize, self.cropsize
            ))

            nn.init.normal_(self.prompt_embeddings.data)

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        # modify first conv layer
        old_weight = model.conv1.weight  # [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(
            self.num_tokens+3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        torch.nn.init.xavier_uniform_(model.conv1.weight)

        model.conv1.weight[:, :3, :, :].data.copy_(old_weight)
        return model

    def incorporate_prompt(self, x):
        batch_size = x.shape[0]
        if self.location == 'pad':
            #batch_size = x.shape[0]
            prompt_emb_lr = self.prompt_norm(self.prompt_embeddings_lr).expand(batch_size, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(self.prompt_embeddings_tb).expand(batch_size, -1, -1, -1)

            x = torch.cat((
                    prompt_emb_lr[:, :, :, :self.num_tokens],
                    x, prompt_emb_lr[:, :, :, self.num_tokens:]), dim=-1)
            x = torch.cat((
                    prompt_emb_tb[:, :, :self.num_tokens, :],
                    x, prompt_emb_tb[:, :, self.num_tokens:, :]), dim=-2)
            #x = self.prompt_layers(x)
        elif self.location == 'below':
            x = torch.cat((
                    x,
                    self.prompt_norm(
                        self.prompt_embeddings).expand(batch_size, -1, -1, -1),
                ), dim=1)
        else:
            raise ValueError("Other initiation scheme is not supported")
        x = self.prompt_layers(x)
        return x

    def get_features(self, x):
        if self.frozen_layers.training:
            self.frozen_layers.eval()
        x = self.incorporate_prompt(x)
        x = self.frozen_layers(x)
        x = self.tuned_layers(x)
        x = x.view(x.size(0), -1)
        return x

    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        '''prompt_layers_keys = ['prompt_layers.'+key for key in self.prompt_layers.state_dict().keys()]
        specific_classes_fc_keys = ['specific_classes_fc.'+key for key in self.specific_classes_fc.state_dict().keys()]
        self.trainable_keys = list(set(prompt_layers_keys).union(specific_classes_fc_keys, grad_keys))'''
        self.trainable_keys = grad_keys

    def init_contorl_parameter_for_scaffold(self, device='cuda'):
        if len(self.trainable_keys) == 0:
            raise ValueError("Forget initializing trainable keys list")
        for key in self.trainable_keys:
            self.control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_control[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)
            self.delta_y[key] = torch.zeros_like(self.state_dict()[key], dtype=torch.float32).to(device)

    def forward(self, x):
        x = self.get_features(x)
        x = self.specific_classes_fc(x)
        return x