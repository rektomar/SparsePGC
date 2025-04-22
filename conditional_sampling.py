import torch
from utils.datasets import MOLECULAR_DATASETS

from utils.conditional import create_conditional_grid
from utils.plot import plot_grid_conditional, plot_grid_unconditional


from rdkit import rdBase
rdBase.DisableLog("rdApp.error")

# nice utility for molecule drawings https://www.rcsb.org/chemical-sketch
# preselected pattern smiles for each dataset
patt_grid_config = {
    'qm9': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC'],
    'zinc250k': ['C1OCC=C1', 'N1NO1', 'CCCO', 'C1CNC1', 'C1=CC=CC=C1', 'C1CN1C', 'N1C=CC=C1', 'COC']
}

patt_eval_config = {
    'qm9': ['COC', 'N1NO1'],
    'zinc250k': ['COC', 'N1NO1']
}

model_path_config = {
    'qm9': 'results/trn/ckpt/qm9/sparse_pgc/dataset=qm9_model=sparse_pgc_order=canonical_nc=256_backend=rtree_vtnl=3_vtnr=10_vtns=32_vtni=32_enl=4_enr=10_ens=32_eni=32_etnl=3_etnr=10_etns=32_etni=32_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=20_batch_size=256_seed=0.pt',
    'zinc250k': 'results/training/model_checkpoint/zinc250k/marg_sort/dataset=zinc250k_model=marg_sort_order=canonical_nc=256_backend=ctree_xnh=256_anh=256_device=cuda_lr=0.05_betas=[0.9, 0.82]_num_epochs=10_batch_size=256_seed=0.pt'
}

if __name__ == "__main__":
    dataset = 'qm9'

    data_info = MOLECULAR_DATASETS[dataset]
    model_path = model_path_config[dataset]
    model = torch.load(model_path, weights_only=False)
    torch.manual_seed(1)

    num_to_sample = 100
    num_to_show = 7  # assuming at least num_to_show of num_samples are valid

    patt_smls = patt_grid_config[dataset]
    cond_smls = create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, data_info)

    # conditional and unconditional sampling grid plots
    plot_grid_conditional(cond_smls, patt_smls, fname=f"{dataset}_cond_mols", useSVG=False)
    plot_grid_unconditional(model, 8, 8, data_info, fname=f"{dataset}_unco_mols", useSVG=False)
