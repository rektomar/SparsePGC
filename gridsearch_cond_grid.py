import os
import torch
import pandas as pd

from utils.datasets import MOLECULAR_DATASETS
from utils.conditional import create_conditional_grid
from utils.plot import plot_grid_conditional

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")

PATT_CONFIG = {
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'CC(C)=O'],
    'zinc250k': ['NS(=O)C1=CC=CC=C1', 'CNC(C)=O', 'O=C1CCCN1', 'C1CCNCC1', 'NS(=O)=O']
}

import re
def get_str_hpar(path, hpar_name):
    match = re.search(rf'{hpar_name}=([^\W_]+)', path)
    return str(match.group(1))

def find_best(evaluation_dir, dataset, model):
    path = evaluation_dir + f'metrics/{dataset}/{model}/'

    def include_name(name):
        # if 'ptree' in name and 'bft' in name:
        #     return True
        # else:
        #     return False
        return True

    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if include_name(f)], ignore_index=True)
    # b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path)], ignore_index=True)
    f_frame = b_frame.loc[b_frame['sam_valid'].idxmax()]
    return f_frame['model_path']

def create_grid(path_model, dataset, num_to_show=8, num_to_sample=1000, seed=0, chunk_size=500, useSVG=False):
    data_info = MOLECULAR_DATASETS[dataset]
    model = torch.load(path_model, weights_only=False)

    model_name = get_str_hpar(path_model, 'model')
    backend_name = get_str_hpar(path_model, 'backend')
    order_name = get_str_hpar(path_model, 'order')

    patt_smls = PATT_CONFIG[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, data_info, seed=seed)

    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_{model_name}_{backend_name}_{order_name}", useSVG=useSVG)


if __name__ == '__main__':
    evaluation_dir = 'results/gs0/eval/'

    # path_model_qm9 = find_best(evaluation_dir, 'qm9', 'marg_sort')
    # print(path_model_qm9)
    # create_grid(path_model_qm9, 'qm9', useSVG=True)

    path_model_zinc250k = find_best(evaluation_dir, 'zinc250k', 'sparse_pgc')
    print(path_model_zinc250k)
    create_grid(path_model_zinc250k, 'zinc250k', useSVG=True)


