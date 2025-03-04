import os
import torch
import urllib
import pandas

from typing import NamedTuple, List, Dict, Optional

from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger

from utils.molecular import mol2sparseg, pad

# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import breadth_first_order, depth_first_order, reverse_cuthill_mckee

BASE_DIR = ''

class MolecularDataset(NamedTuple):
    dataset: str
    max_atoms: int
    max_bonds: int
    atom_list: List[int]
    valency_dict: Optional[Dict[int, int]] = None

# TODO: add valencies
MOLECULAR_DATASETS = {
    'qm9': MolecularDataset('qm9', 9, 13, 
                            [0, 6, 7, 8, 9], 
                            {6: 4, 7: 3, 8: 2, 9: 1}),
    'zinc250k': MolecularDataset('zinc250k', 38, 45, 
                                 [0, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                                 {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}),
    'moses': MolecularDataset('moses', 27, 31, 
                              [0, 6, 7, 8, 9, 16, 17, 35], 
                              {6: 4, 7: 3, 8: 2, 9: 1, 16: 2, 17: 1, 35: 1}),
    'guacamol': MolecularDataset('guacamol', 88, 87,  
                                 [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]),
    'polymer': MolecularDataset('polymer', 122, 145, 
                                [0, 6, 7, 8, 9, 14, 15, 16])
}


# Moses Atoms - C:6, N:7, S:16, O:8, F:9, Cl:17, Br:35, H:1
# Guacamol Atoms - C:6, N:7, O:8, F:9, B:5, Br:35, Cl:17, I:53, P:15, S:16, Se:34, Si:14, H:1

def download_qm9(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}qm9'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print('Downloading and preprocessing the QM9 dataset.')

    if not os.path.exists(f'{file}.csv'):
        urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', MOLECULAR_DATASETS['qm9'], order)
    # os.remove(f'{file}.csv')

    print('Done.')

def download_zinc250k(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}zinc250k'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc250k_property.csv'

    print('Downloading and preprocessing the Zinc250k dataset.')

    if not os.path.exists(f'{file}.csv'):
        urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', MOLECULAR_DATASETS['zinc250k'], order)
    # os.remove(f'{file}.csv')

    print('Done.')

def download_moses(dir='data/', order='canonical'):
    # https://github.com/cvignac/DiGress/blob/main/src/datasets/moses_dataset.py
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}moses'
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    test_url  = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    # test_url  = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    # NOTE: Downloading just train split so far.
    if not os.path.exists(f'{file}.csv'):
        urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, 'SMILES', MOLECULAR_DATASETS['moses'], order)
    # os.remove(f'{file}.csv')

    print('Done.')

def download_guacamol(dir='data/', order='canonical'):
    # https://github.com/cvignac/DiGress/blob/main/src/datasets/guacamol_dataset.py
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}guacamol'
    train_url = 'https://figshare.com/ndownloader/files/13612760'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'
    test_url = 'https://figshare.com/ndownloader/files/13612757'

    # NOTE: Downloading just train split so far.
    if not os.path.exists(f'{file}.csv'):
        urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, None, MOLECULAR_DATASETS['guacamol'], order)
    # os.remove(f'{file}.csv')

    print('Done.')

def download_polymer(dir='data/', order='canonical'):
    # https://github.com/wengong-jin/hgraph2graph/tree/master/data/polymers
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}polymer'
    train_url = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/train.txt'
    valid_url = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/valid.txt'
    test_url  = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/test.txt'

    # NOTE: Downloading just train split so far.
    if not os.path.exists(f'{file}.csv'):
        urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, None, MOLECULAR_DATASETS['polymer'], order)
    # os.remove(f'{file}.csv')

    print('Done.')
    
def preprocess(path, smile_col, data_info, order='canonical'):
    if smile_col is not None:
        input_df = pandas.read_csv(f'{path}.csv', sep=',', dtype='str')
        smls_list = list(input_df[smile_col])
    else:
        input_df = pandas.read_csv(f'{path}.csv', header=None, sep=',', dtype='str')
        smls_list = list(input_df[0])

    data_list = []

    for sml in tqdm(smls_list):
        mol = Chem.MolFromSmiles(sml)
        Chem.Kekulize(mol)
        n = mol.GetNumAtoms()
        m = mol.GetNumBonds()

        if m < 1:
            print(f'Skipping {sml}')
            continue
        atom_tensor, bond_tensor = mol2sparseg(mol, data_info)

        data_list.append({'s': sml, 'v': atom_tensor, 'e': bond_tensor, 'n': n, 'm': m})

    torch.save(data_list, f'{path}_{order}.pt')

def valency_analysis(name, order='canonical'):
    valencies = set()
    data = torch.load(f'data/{name}_{order}.pt', weights_only=True)
    for datapoint in data:
        sml = datapoint['s']
        mol = Chem.MolFromSmiles(sml)
        Chem.Kekulize(mol)

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            valency = atom.GetTotalValence()

            valencies.add((atomic_num, valency)) 
    print(f'{name} valencies: {valencies}')
    return valencies


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def collate_dict(batch, data_info):
    out = {}
    out['s'] = [d['s'] for d in batch]

    out['v'] = torch.stack([pad(d['v'], data_info.max_atoms) for d in batch], 0)
    out['e'] = torch.stack([pad(d['e'], data_info.max_bonds) for d in batch], 0)

    out['n'] = torch.tensor([d['n'] for d in batch])
    out['m'] = torch.tensor([d['m'] for d in batch])
    return out

def load_dataset(name, batch_size, split, seed=0, dir='data/', order='canonical'):
    x = DictDataset(torch.load(f'{dir}{name}_{order}.pt', weights_only=True))

    def collate_wrapper(batch):
        return collate_dict(batch, MOLECULAR_DATASETS[name])

    torch.manual_seed(seed)
    x_trn, x_val, x_tst = torch.utils.data.random_split(x, split)

    loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=False, collate_fn=collate_wrapper, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, collate_fn=collate_wrapper, pin_memory=True)
    loader_tst = torch.utils.data.DataLoader(x_tst, batch_size=batch_size, num_workers=2, shuffle=False, collate_fn=collate_wrapper, pin_memory=True)

    smiles_trn = [x['s'] for x in loader_trn.dataset]
    smiles_val = [x['s'] for x in loader_val.dataset]
    smiles_tst = [x['s'] for x in loader_tst.dataset]

    return {
        'loader_trn': loader_trn,
        'loader_val': loader_val,
        'loader_tst': loader_tst,
        'smiles_trn': smiles_trn,
        'smiles_val': smiles_val,
        'smiles_tst': smiles_tst
    }


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    torch.set_printoptions(threshold=10_000, linewidth=200)

    download = True
    datasets = ['qm9', 'zinc250k', 'moses', 'guacamol', 'polymer']
    # datasets = ['guacamol', 'polymer']
    orders = ['canonical']

    for dataset in datasets:
        for order in orders:
            if download:
                match dataset:
                    case 'qm9':
                        download_qm9(order=order)
                    case 'zinc250k':
                        download_zinc250k(order=order)
                    case 'moses':
                        download_moses(order=order)
                    case 'guacamol':
                        download_guacamol(order=order)
                    case 'polymer':
                        download_polymer(order=order)
                    case _:
                        os.error('Unsupported dataset.')

            loaders = load_dataset(dataset, 100, split=[0.8, 0.1, 0.1], order=order)

    # valency_analysis('zinc250k')
