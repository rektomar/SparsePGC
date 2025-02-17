import torch

def permute_graph(xx, aa, pi):
    px = xx[pi, ...]
    pa = aa[pi, :, ...]
    pa = pa[:, pi, ...]
    return px, pa

def flatten_tril(a, max_atom):
    m = torch.tril(torch.ones(max_atom, max_atom, dtype=torch.bool), diagonal=-1)
    return a[..., m].reshape(-1)

def unflatt_tril(l, max_atom):
    m = torch.tril(torch.ones(max_atom, max_atom, dtype=torch.bool), diagonal=-1)
    a = torch.zeros(*l.shape[:-1], max_atom, max_atom).type_as(l)
    a[..., m] = l
    a.transpose(1, 2)[..., m] = l
    return a
