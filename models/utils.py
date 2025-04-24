import torch
import torch.nn as nn
from torch.distributions import Categorical


class Cardinality(nn.Module):
    def __init__(self, max_atoms: int, max_bonds: int, device='cuda'):
        super().__init__()
        self.max_atoms = max_atoms
        self.max_bonds = max_bonds

        self.logits = nn.Parameter(torch.randn(max_atoms, max_bonds, device=device), requires_grad=True)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, n: torch.Tensor, m: torch.Tensor):
        logs = self.logits.view(-1).log_softmax(-1).view(*self.logits.shape)
        return logs[n, m]

    @torch.no_grad
    def sample(self, num_samples: int):
        logs = self.logits.view(-1).log_softmax(-1)
        indices = Categorical(logits=logs).sample((num_samples,))
        n, m = torch.div(indices, self.logits.shape[1], rounding_mode='floor'), indices % self.logits.shape[1]
        return n, m
    
    @torch.no_grad
    def sample_conditional(self, n: torch.Tensor, m: torch.Tensor):
        assert len(n) == len(m)
        atom_mask = torch.arange(self.max_atoms, device=self.device).view(1, -1) <= n.view(-1, 1)
        bond_mask = torch.arange(self.max_bonds, device=self.device).view(1, -1) <= m.view(-1, 1) 

        # atom_mask to [bs, max_atoms, 1]
        # bond_mask to [bs, 1, max_bonds]
        mask = atom_mask.unsqueeze(2) | bond_mask.unsqueeze(1)
        logits = self.logits.unsqueeze(0).expand(mask.shape[0], -1, -1).clone()
        logits[mask] = -torch.inf
        logs = logits.view(-1, self.max_atoms*self.max_bonds).log_softmax(-1)
        indices = Categorical(logits=logs).sample()
        n_cond, m_cond = torch.div(indices, self.logits.shape[1], rounding_mode='floor'), indices % self.logits.shape[1]
        return n_cond, m_cond

if __name__ == "__main__":
    model = Cardinality(7, 11)

    n = torch.tensor([0, 1, 2, 3], device='cuda')
    m = torch.tensor([0, 1, 2, 3], device='cuda')

    n_cond, m_cond = model.sample_conditional(n, m)
    print(n_cond, m_cond)