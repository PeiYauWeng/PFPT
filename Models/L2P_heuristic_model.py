import torch
from torch import nn
import torchvision.models as models
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from collections import Counter, OrderedDict

class Pool(nn.Module):
    def __init__(self, patch_size=32, embed_dim=768, embedding_key='cls', prompt_init='uniform',
                 pool_size=None, top_k=None, batchwise_prompt=False,  dropout_value=0.0):
        super(Pool, self).__init__()
        patch_size_pair = _pair((patch_size, patch_size))
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.top_k_idx = torch.zeros(top_k)
        self.batchwise_prompt = batchwise_prompt
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.features_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.features_dropout = nn.Dropout(dropout_value)
        if prompt_init == 'uniform':
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size_pair, 1) + embed_dim))
            nn.init.uniform_(self.prompt.data, -val, val)
            nn.init.uniform_(self.features_proj.weight, -1, 1)
        else:
            raise NotImplementedError("Not supported way of prompt initial!")
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
        
    def forward(self, x_embed, cls_features=None, used_prompts_frequency=None):
        current_pool_size = self.prompt.shape[0]
        if self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
            
        prompt_norm = self.l2_normalize(self.prompt, dim=1) # Pool_size, C
        x_embed_mean = self.features_proj(self.features_dropout(x_embed_mean))
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < current_pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((current_pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((current_pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
            major_prompt_id = prompt_id[major_idx] # top_k
            # expand to batch
            idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            self.top_k_idx = idx[0]
        
        
        batched_prompt = self.prompt[idx] # B, top_k, C
        if used_prompts_frequency is not None and torch.sum(used_prompts_frequency)>0:
            used_prompts_frequency = 1 - (used_prompts_frequency/torch.max(used_prompts_frequency))
            weighted_prompt_norm = prompt_norm * used_prompts_frequency.unsqueeze(1).to(prompt_norm.device)
            batched_key_norm = weighted_prompt_norm[idx] # B, top_k, C
        else:
            batched_key_norm = prompt_norm[idx]
        
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0]
        
        return reduce_sim, batched_prompt
        
        
class L2P_ViT_B32(nn.Module):
    def __init__(self, prompt_method, batchwise_prompt, classification_adaptor=True,
                 pool_size=None, top_k=None, frozen_pretrian=True, num_classes=10):
        super(L2P_ViT_B32, self).__init__()
        self.num_classes = num_classes
        self.trainable_keys = list()
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
        self.pool = Pool(embed_dim=hidden_size_each_layers[0], pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt)
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
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
        
    def get_cls_features(self, x):
        with torch.no_grad():
            x = self.vit_b32._process_input(x)
            n = x.shape[0]
            batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.vit_b32.encoder(x)
            x = x[:, 0]
        return x
        
    def forward_features(self, x, cls_features):
        x = self.vit_b32._process_input(x)
        n = x.shape[0]
        reduce_sim, batched_prompt = self.pool(x, cls_features=cls_features)
        batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x += self.vit_b32.encoder.pos_embedding
        x = self.vit_b32.encoder.dropout(x)
        x = torch.cat((
            x[:, :1, :],
            batched_prompt,
            x[:, 1:, :]
        ), dim=1)
        x = self.vit_b32.encoder.dropout(x)
        x = self.vit_b32.encoder.layers(x)
        features = self.vit_b32.encoder.ln(x)
        return reduce_sim, features
        
    def forward_head(self, features):
        x = features[:, 0]
        x = self.vit_b32.heads(x)
        x = self.classification_head(x)
        return x
        
    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad == True:
                grad_keys.append(n)
        self.trainable_keys = grad_keys

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx] += 1.0
        
    def reset_trained_pormpts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.torch.float32)
        
    def forward(self, x):
        cls_features = self.get_cls_features(x)
        reduce_sim, features = self.forward_features(x, cls_features=cls_features)
        logits = self.forward_head(features)
        self.checking_trained_prompt()
        return reduce_sim, logits
        
        
        
        
        
        

