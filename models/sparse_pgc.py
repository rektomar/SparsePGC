import torch
import torch.nn as nn

from models.backend import backend_selector
from typing import Optional


class SparsePGC(nn.Module):
    def __init__(self, dataset, hpars):
        super().__init__()

        network_vt, network_e, network_et, max_atoms, max_bonds = backend_selector(dataset, hpars)

        self.nc = hpars['nc']

        self.network_vtype = network_vt
        self.network_edges = network_e
        self.network_etype = network_et

        self.max_atoms = max_atoms
        self.max_bonds = max_bonds

        self.logits_n = nn.Parameter(torch.randn(max_atoms, device=self.device), requires_grad=True)
        self.logits_m = nn.Parameter(torch.randn(max_bonds+1, device=self.device), requires_grad=True)
        self.logits_w = nn.Parameter(torch.randn(self.nc,   device=self.device), requires_grad=True)

        self.to(hpars['device'])
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, v: torch.Tensor, e: torch.Tensor):

        vtype = v[..., 1]
        edges, etype = e[..., :2], e[..., 2]
        edges = torch.flatten(edges, start_dim=1)

        mask_vtype = vtype != -1
        mask_edges = edges != -1
        mask_etype = etype != -1

        self.network_vtype.set_marginalization_mask(mask_vtype)
        self.network_edges.set_marginalization_mask(mask_edges)
        self.network_etype.set_marginalization_mask(mask_etype)

        n = mask_vtype.sum(dim=1) - 1
        m = mask_etype.sum(dim=1) - 1
        dist_n = torch.distributions.Categorical(logits=self.logits_n)
        dist_m = torch.distributions.Categorical(logits=self.logits_m)


        logs_n = dist_n.log_prob(n)
        logs_m = dist_m.log_prob(m)
        logs_vtype = self.network_vtype(vtype)
        logs_edges = self.network_edges(edges)
        logs_etype = self.network_etype(etype)
        logs_w = torch.log_softmax(self.logits_w, dim=0).unsqueeze(0)

        return logs_n + logs_m + torch.logsumexp(logs_vtype + logs_edges + logs_etype + logs_w, dim=1)

    def logpdf(self, v: torch.Tensor, e: torch.Tensor):
        return self(v, e).mean()
    
    def sample(self, num_samples: int):
        samp_n = torch.distributions.Categorical(logits=self.logits_n).sample((num_samples, ))
        samp_m = torch.distributions.Categorical(logits=self.logits_m).sample((num_samples, ))

        samp_w = torch.distributions.Categorical(logits=self.logits_w).sample((num_samples, ))

        vtype = self.network_vtype.sample(num_samples, class_idxs=samp_w)
        edges = self.network_edges.sample(num_samples, class_idxs=samp_w)
        etype = self.network_etype.sample(num_samples, class_idxs=samp_w)

        mask_v = torch.arange(self.max_atoms, device=self.device).unsqueeze(0) <= samp_n.unsqueeze(1)
        ids = torch.arange(self.max_atoms, device=self.device)
        ids = ids.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, -1)
        v = torch.cat((ids, vtype.unsqueeze(-1)), -1)
        v[~mask_v] = -1

        mask_e = torch.arange(self.max_bonds, device=self.device).unsqueeze(0) <= samp_m.unsqueeze(1)
        edges = edges.view(edges.shape[0], -1, 2)
        e = torch.cat((edges, etype.unsqueeze(-1)), -1)
        e[~mask_e] = -1

        return v, e

MODELS = {
    'sparse_pgc': SparsePGC,
}

if __name__ == '__main__':
    import json
    from utils.datasets import load_dataset

    dataset = 'qm9'

    with open(f'config/{dataset}/spgc_btree.json', 'r') as f:
        hyperpars = json.load(f)
    hyperpars['model_hpars']['device'] = 'cpu'

    loaders = load_dataset(dataset, 256, split=[0.8, 0.1, 0.1], order='canonical')
    model = SparsePGC(dataset, hyperpars['model_hpars'])
    print(model)

    batch = next(iter(loaders['loader_trn']))
    ll = model(batch['v'], batch['e'])
    print(ll.shape)
    model.sample(16)
