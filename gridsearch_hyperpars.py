import json
import itertools

from utils.templates_hyperpars import *


def grid_btree(nl, ns, ni):
    return [template_shpgc_btree(*p) for p in list(itertools.product(nl, ns, ni))]

def grid_rtree(nl, nr, ns, ni):
    return [template_shpgc_rtree(*p) for p in list(itertools.product(nl, nr, ns, ni))]

def grid_ptree(nl, ns, ni):
    return [template_shpgc_ptree(*p) for p in list(itertools.product(nl, ns, ni))]


def grid_spgc(dataset, model):
    order = ['canonical', 'bft']
    nc = [512, 256, 128]
    backend_name = ['btree', 'rtree', 'ptree']
    backend_grid = [grid_btree, grid_rtree, grid_ptree]
    match dataset:
        case 'qm9':
            backend_vtpar = [
                {"nl":[3],            "ns":[32], "ni":[32]},
                {"nl":[3], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[3],            "ns":[32], "ni":[32]},
            ]
            backend_epar = [
                {"nl":[5],            "ns":[32], "ni":[32]},
                {"nl":[5], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[5],            "ns":[32], "ni":[32]},
            ]
            backend_etpar = [
                {"nl":[5],            "ns":[32], "ni":[32]},
                {"nl":[5], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[5],            "ns":[32], "ni":[32]},
            ]
        case 'zinc250k':
            backend_vtpar = [
                {"nl":[4],            "ns":[32], "ni":[32]},
                {"nl":[4], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[4],            "ns":[32], "ni":[32]},
            ]
            backend_epar = [
                {"nl":[6],            "ns":[32], "ni":[32]},
                {"nl":[6], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[6],            "ns":[32], "ni":[32]},
            ]
            backend_etpar = [
                {"nl":[6],            "ns":[32], "ni":[32]},
                {"nl":[6], "nr":[16], "ns":[32], "ni":[32]},
                {"nl":[6],            "ns":[32], "ni":[32]},
            ]
        case _:
            raise 'Unknown dataset'
    backend_nr = [
        [None],
        [None],
        [16]
        ]
    batch_size = [256]
    lr = [0.01, 0.05, 0.1]
    seed = [0, 1, 2, 3, 4]

    hyperpars = []
    for b_name, b_grid, b_vtpar, b_epar, b_etpar, b_nr in zip(backend_name, backend_grid, backend_vtpar, backend_epar, backend_etpar, backend_nr):
        grid = itertools.product(order, nc, b_nr, [b_name], b_grid(**b_vtpar), b_grid(**b_epar), b_grid(**b_etpar), batch_size, lr, seed)
        hyperpars.extend([template_sort(dataset, model, *p) for p in list(grid)])

    return hyperpars


GRIDS = {
    'sparse_pgc': grid_spgc,
}


if __name__ == "__main__":
    # print(len(grid_spgc('qm9', 'sparse_pgc')))
    for p in grid_spgc('qm9', 'sparse_pgc')[-9:]:
        print(json.dumps(p, indent=4))