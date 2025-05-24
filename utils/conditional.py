import torch

from rdkit import Chem, rdBase
rdBase.DisableLog("rdApp.error")

from utils.molecular import mol2sparseg, sparseg2mol, sparsegs2mols, isvalid, pad
from utils.evaluate import validate_sparsegs

def sample_conditional(model, v, e, data_info):

    vc, ec = model.sample_conditional(v, e)
    vc, ec = validate_sparsegs(vc, ec)
    mol_sample = sparsegs2mols(vc, ec, data_info)
    sml_sample = [Chem.MolToSmiles(mol, kekuleSmiles=True) for mol in mol_sample]

    return vc, ec, mol_sample, sml_sample

def create_observed_mol(smile, data_info, order='canonical', device='cuda'):
    assert order == 'canonical'
    mol = Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol)
    v, e = mol2sparseg(mol, data_info)
    # x, a, mol, s = reorder_molecule(x, a, mol, order, max_atom, atom_list)

    v, e = pad(v, data_info.max_atoms), pad(e, data_info.max_bonds)
    v, e = v.unsqueeze(0).float().to(device), e.unsqueeze(0).float().to(device)
    
    return v, e

def create_conditional_grid(model, patt_smls, num_to_show, num_to_sample, data_info, seed=0):
    assert num_to_show < num_to_sample
    cond_smls = []

    for patt in patt_smls:
        vo, eo = create_observed_mol(patt, data_info)
        vo = vo.expand(num_to_sample, -1, -1).clone()
        eo = eo.expand(num_to_sample, -1, -1).clone()
        torch.manual_seed(seed)
        _, _, mols, smls = sample_conditional(model, vo, eo, data_info)
        filtered = [(mol, sml) for (mol, sml) in zip(mols, smls) if isvalid(mol)]
        if len(filtered) == 0:
            valid_mols, valid_smls = [], []
        else:
            valid_mols, valid_smls = zip(*filtered)
            valid_mols, valid_smls = list(valid_mols), list(valid_smls)

        # TODO: num_to_show > num_valid case
        final_smls = valid_smls[:num_to_show]
        print(f"Pattern {patt}: {final_smls}")

        cond_smls.append(final_smls)
    return cond_smls
