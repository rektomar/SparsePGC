import os

from utils.datasets import MOLECULAR_DATASETS, BOND_DECODER
from models.einsum import Graph, EinsumNetwork, ExponentialFamilyArray

class BTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 nd,
                 nk,
                 nc,
                 nl,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.Args(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)
        graph = Graph.binary_tree(nd, nl, 'half')

        super().__init__(graph, args)
        self.initialize()

class RTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 nd,
                 nk,
                 nc,
                 nl,
                 nr,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.Args(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=nc,
            exponential_family=ExponentialFamilyArray.CategoricalArray,
            exponential_family_args={'K': nk},
            use_em=False)
        graph = Graph.random_binary_trees(nd, nl, nr)

        super().__init__(graph, args)
        self.initialize()


def backend_selector(dataset, hpars):
    data_info = MOLECULAR_DATASETS[dataset]

    nd_vt, nk_vt = data_info.max_atoms, len(data_info.atom_list)
    nd_e, nk_e   = 2*data_info.max_bonds, data_info.max_atoms 
    nd_et, nk_et = data_info.max_bonds, len(BOND_DECODER)

    nc = hpars['nc']

    match hpars['backend']:
        case 'btree':
            network_vt = BTreeSPN(   nd_vt, nk_vt, nc, **hpars['bvt_hpars'])
            network_e  = BTreeSPN(   nd_e,  nk_e,  nc, **hpars['be_hpars'])
            network_et = BTreeSPN(   nd_et, nk_et, nc, **hpars['bet_hpars'])
        # case 'rtree':
        #     network_x = RTreeSPN(   nd_x, nk_x, nc, **hpars['bx_hpars'])
        #     network_a = RTreeSPN(   nd_a, nk_a, nc, **hpars['ba_hpars'])
        case _:
            os.error('Unknown backend')

    return network_vt, network_e, network_et, data_info.max_atoms, data_info.max_bonds


if __name__ == '__main__':
    dataset = 'qm9'

    import json

    with open(f'config/{dataset}/spgc_btree.json', 'r') as f:
        hyperpars = json.load(f)

    network_vt, network_e, network_et, _, _ = backend_selector(dataset, hyperpars['model_hpars'])
    print(network_vt, network_e, network_et)
