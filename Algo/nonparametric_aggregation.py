import torch
import torch.nn as nn
from torch.optim import Adam
import copy
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from scipy.optimize import linear_sum_assignment

class NonparametricAgg(nn.Module):
    def __init__(self, prompt_dim, n_hidden=128):
        super(NonparametricAgg, self).__init__()
        self.cov_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, prompt_dim),
            nn.Sigmoid()
        )
        self.bernoulli_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def prompt_likelihood(self, local_prompts, centroids, z):
        _z = torch.tensor(z).to(local_prompts.device)
        lik = 0.
        cost_mat = []
        for i in range(centroids.shape[0]):
            mean_i = centroids[i].view(1, -1)
            cov_i = self.cov_net(mean_i)
            prompt_dist = Independent(Normal(mean_i, cov_i),1)
            lp = prompt_dist.log_prob(local_prompts)
            cost_mat.append(lp)
            log_prob = _z[:, i] * cost_mat[-1] # n_local
            lik += log_prob.sum()
        return lik, torch.stack(cost_mat) # n_global x n_local

    def z_likelihood(self, centroids, z):
        _z = torch.tensor(z).to(centroids.device)
        c = torch.sum(_z, dim=0) # n_global
        lik = 0.
        cost_mat = []
        for i in range(centroids.shape[0]):
            prob_i = self.bernoulli_net(centroids[i].view(1, -1))
            prompt_dist = Independent(Bernoulli(prob_i),1)
            cost_mat.append(prompt_dist.log_prob(c[i] * torch.ones(_z.shape[0]).to(centroids.device)))
            log_prob = _z[:, i] * cost_mat[-1]   # n_local
            lik += log_prob.sum()
        return lik, torch.stack(cost_mat) # n_global x n_local

    # local_prompts: n_clients x n_prompts x 768
    def forward(self, local_prompts, outer_loop=50):
        n_clients, n_local = local_prompts.shape[0], local_prompts.shape[1]
        n_global = n_clients * n_local
        # Initialize z
        z = []
        for i in range(n_clients):
            perm = np.arange(n_global)
            np.random.shuffle(perm)
            zi = np.zeros((n_local, n_global))
            for j in range(n_local):
                zi[j][perm[j]] = 1
            z.append(zi)

        centroids = nn.ParameterList([copy.deepcopy(local_prompts.flatten(0, 1))]) # (n_clients x n_prompts) x 768
        opt = Adam([
            {'params': self.cov_net.parameters()},
            {'params': self.bernoulli_net.parameters()},
            {'params': centroids}
        ])

        # Alternate opt phi, z
        for i in range(outer_loop):
            for t in range(n_clients):
                opt.zero_grad()
                # Compute l1, l2
                l1, m1 = self.prompt_likelihood(local_prompts[t], centroids[0], z[t])
                l2, m2 = self.z_likelihood(centroids[0], z[t])
                #loss.append(l1 + l2)
                loss = -l1 -l2
                loss.backward()
                opt.step()

                # Solve for z
                m = (m1 + m2).t().detach().cpu().numpy()
                row_id, col_id = linear_sum_assignment(m, maximize=True)
                z[t] *= 0
                z[t][row_id, col_id] += 1

            #loss = torch.stack(loss).sum()
            #loss.backward()
            #opt.step()
        z = np.stack(z)
        z = np.sum(np.stack(z), axis=(0, 1), keepdims=False) # n_local x n_global
        global_prompts = centroids[0][np.where(z > 0)[0]]
        del z, centroids
        return global_prompts
