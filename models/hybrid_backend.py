import os

from utils.datasets import MOLECULAR_DATASETS
from utils.molecular import BOND_DECODER
from models.hybrid_einsum import Graph, EinsumNetwork, ExponentialFamilyArray

class BTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 exponential_family_args,
                 nd,
                 nl,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.HybridArgs(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=1,
            exponential_family_args=exponential_family_args,
            use_em=False)
        graph = Graph.binary_tree(nd, nl, 'half')

        super().__init__(graph, args)
        self.initialize()

class RTreeSPN(EinsumNetwork.EinsumNetwork):
    def __init__(self,
                 exponential_family_args,
                 nd,
                 nl,
                 nr,
                 ns,
                 ni
                 ):
        args = EinsumNetwork.HybridArgs(
            num_var=nd,
            num_dims=1,
            num_input_distributions=ni,
            num_sums=ns,
            num_classes=1,
            exponential_family_args=exponential_family_args,
            use_em=False)
        graph = Graph.random_binary_trees(nd, nl, nr)

        super().__init__(graph, args)
        self.initialize()


def hybrid_backend_selector(dataset, hpars):
    data_info = MOLECULAR_DATASETS[dataset]

    nd_vt, nk_vt = data_info.max_atoms, len(data_info.atom_list)
    nd_e, nk_e   = 2*data_info.max_bonds, data_info.max_atoms 
    nd_et, nk_et = data_info.max_bonds, len(BOND_DECODER)

    eponential_family_args = [
        (ExponentialFamilyArray.CategoricalArray, {'K': nk_vt}, range(0, nd_vt)),
        (ExponentialFamilyArray.CategoricalArray, {'K': nk_e }, range(nd_vt, nd_vt+nd_e)),
        (ExponentialFamilyArray.CategoricalArray, {'K': nk_et}, range(nd_vt+nd_e, nd_vt+nd_e+nd_et))
    ]

    nd = nd_vt+nd_e+nd_et

    match hpars['backend']:
        case 'btree':
            network = BTreeSPN(eponential_family_args, nd, **hpars['b_hpars'])
        case 'rtree':
            network = RTreeSPN(eponential_family_args, nd, **hpars['b_hpars'])
        case _:
            os.error('Unknown backend')

    return network, data_info.max_atoms, data_info.max_bonds

if __name__ == '__main__':
    dataset = 'qm9'

    import json

    with open(f'config/{dataset}/shpgc_rtree.json', 'r') as f:
        hyperpars = json.load(f)

    network, _, _ = hybrid_backend_selector(dataset, hyperpars['model_hpars'])
    print(network)
