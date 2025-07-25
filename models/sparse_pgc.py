import torch
import torch.nn as nn

from models.backend import backend_selector
from models.utils import Cardinality

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

        self.network_card = Cardinality(self.max_atoms, self.max_bonds, device=hpars['device'])

        self.logits_w = nn.Parameter(torch.randn(self.nc,   device=hpars['device']), requires_grad=True)

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

        logs_card  = self.network_card(n, m)
        logs_vtype = self.network_vtype(vtype)
        logs_edges = self.network_edges(edges)
        logs_etype = self.network_etype(etype)
        logs_w = torch.log_softmax(self.logits_w, dim=0).unsqueeze(0)

        return logs_card + torch.logsumexp(logs_vtype + logs_edges + logs_etype + logs_w, dim=1)

    def logpdf(self, v: torch.Tensor, e: torch.Tensor):
        return self(v, e).mean()
    
    def _sample(self, num_samples: int):
        samp_n, samp_m = self.network_card.sample(num_samples)

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
    
    @torch.no_grad
    def sample(self, num_samples: int, chunk_size: int=1000):
        v_sam = []
        e_sam = []

        if num_samples > chunk_size:
            chunks = num_samples // chunk_size*[chunk_size] + ([num_samples % chunk_size] if num_samples % chunk_size > 0 else [])
            for n in chunks:
                v, e = self._sample(n)
                v_sam.append(v)
                e_sam.append(e)
            v_sam, e_sam = torch.cat(v_sam), torch.cat(e_sam)
        else:
            v_sam, e_sam = self._sample(num_samples)  

        return v_sam, e_sam

    @torch.no_grad
    def sample_conditional(self, v_cond: torch.tensor, e_cond: torch.tensor, n=None, m=None):

        vtype_o = v_cond[..., 1]
        edges_o, etype_o = e_cond[..., :2], e_cond[..., 2]
        edges_o = torch.flatten(edges_o, start_dim=1)

        mask_vtype = vtype_o != -1
        mask_edges = edges_o != -1
        mask_etype = etype_o != -1
        
        if m is not None and n is not None:
            samp_n, samp_m = n, m
        else:
            n_o = mask_vtype.sum(dim=1) - 1
            m_o = mask_etype.sum(dim=1) - 1
            samp_n, samp_m = self.network_card.sample_conditional(n_o, m_o)

        num_samples = len(v_cond)

        self.network_vtype.set_marginalization_mask(mask_vtype)
        self.network_edges.set_marginalization_mask(mask_edges)
        self.network_etype.set_marginalization_mask(mask_etype)

        ll_vtype = self.network_vtype(vtype_o)
        ll_edges = self.network_edges(edges_o)
        ll_etype = self.network_etype(etype_o)

        logits_w = self.logits_w.unsqueeze(0) + ll_vtype + ll_edges + ll_etype
        samp_w = torch.distributions.Categorical(logits=logits_w).sample()

        vtype = self.network_vtype.sample(num_samples, class_idxs=samp_w, x=vtype_o)
        edges = self.network_edges.sample(num_samples, class_idxs=samp_w, x=edges_o)
        etype = self.network_etype.sample(num_samples, class_idxs=samp_w, x=etype_o)

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
    # hyperpars['model_hpars']['device'] = 'cpu'

    loaders = load_dataset(dataset, 256, split=[0.8, 0.1, 0.1], order='canonical')
    model = SparsePGC(dataset, hyperpars['model_hpars'])
    print(model)

    batch = next(iter(loaders['loader_trn']))
    ll = model(batch['v'].to(model.device), batch['e'].to(model.device))
    print(ll.shape)
    model.sample(16)
