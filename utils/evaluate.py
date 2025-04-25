import torch
from rdkit import Chem
from utils.molecular import mols2sparsegs, sparsegs2mols, mols2smls, get_vmols, validate_sparsegs

from utils.metrics.nspdk import metric_nspdk
# from utils.metrics.kldiv import metric_k
from utils.metrics.fcd import metric_f 


def metric_v(vmols, num_mols):
    return len(vmols) / num_mols

def metric_u(vsmls, num_mols):
    num_umols = len(list(set(vsmls)))
    if num_umols == 0:
        return 0., 0.
    else:
        return num_umols / len(vsmls), num_umols / num_mols

def metric_n(vsmls, tsmls, num_mols):
    usmls = list(set(vsmls))
    num_umols = len(usmls)
    if num_umols == 0:
        return 0., 0.
    else:
        num_nmols = num_umols - sum([1 for mol in usmls if mol in tsmls])
        return num_nmols / num_umols, num_nmols / num_mols

def metric_m(ratio_v, ratio_u, ratio_n):
    return ratio_v*ratio_u*ratio_n

def metric_s(mols, num_mols):
    mols_stable = 0
    bond_stable = 0
    sum_atoms = 0

    for mol in mols:
        mol = Chem.AddHs(mol, explicitOnly=True)
        num_atoms = mol.GetNumAtoms()
        num_stable_bonds = 0
        for atom in mol.GetAtoms():
            num_stable_bonds += int(atom.HasValenceViolation() == False)

        mols_stable += int(num_stable_bonds == num_atoms)
        bond_stable += num_stable_bonds
        sum_atoms += num_atoms

    return mols_stable / float(num_mols), bond_stable / float(sum_atoms)

def evaluate_molecules(
        v,
        e,
        loaders,
        data_info,
        evaluate_trn=False,
        evaluate_val=False,
        evaluate_tst=False,
        metrics_only=False,
        canonical=True,
        preffix='',
        device="cuda"
    ):
    num_mols = len(v)

    mols = sparsegs2mols(v, e, data_info)
    smls = mols2smls(mols, canonical)
    vmols, vsmls = get_vmols(smls)

    ratio_v = metric_v(vmols, num_mols)
    ratio_u, ratio_u_abs = metric_u(vsmls, num_mols)
    ratio_n, ratio_n_abs = metric_n(vsmls, loaders['smiles_trn'], num_mols)
    ratio_s = metric_m(ratio_v, ratio_u, ratio_n)
    ratio_m, ratio_a = metric_s(mols, num_mols)

    metrics = {
        f'{preffix}valid': ratio_v,
        f'{preffix}unique': ratio_u,
        f'{preffix}unique_abs': ratio_u_abs,
        f'{preffix}novel': ratio_n,
        f'{preffix}novel_abs': ratio_n_abs,
        f'{preffix}score': ratio_s,
        f'{preffix}m_stab': ratio_m,
        f'{preffix}a_stab': ratio_a
    }

    if evaluate_trn == True:
        metrics = metrics | {
            f'{preffix}fcd_trn'  : metric_f(vsmls, loaders['smiles_trn'], device, canonical),
            # f'{preffix}kldiv_trn': metric_k(vsmls, loaders['smiles_trn']),
            # f'{preffix}nspdk_trn': metric_nspdk(vsmls, loaders['smiles_trn']),
        }
        print('Finished evaluating trn set')
    if evaluate_val == True:
        metrics = metrics | {
            f'{preffix}fcd_val'  : metric_f(vsmls, loaders['smiles_val'], device, canonical),
            # f'{preffix}kldiv_val': metric_k(vsmls, loaders['smiles_val']),
            f'{preffix}nspdk_val': metric_nspdk(vsmls, loaders['smiles_val']),
        }
        print('Finished evaluating val set')
    if evaluate_tst == True:
        metrics = metrics | {
            f'{preffix}fcd_tst'  : metric_f(vsmls, loaders['smiles_tst'], device, canonical),
            # f'{preffix}kldiv_tst': metric_k(vsmls, loaders['smiles_tst']),
            f'{preffix}nspdk_tst': metric_nspdk(vsmls, loaders['smiles_tst']),
        }
        print('Finished evaluating tst set')

    if metrics_only == True:
        return metrics
    else:
        return vmols, vsmls, metrics

def print_metrics(metrics):
    return f'v={metrics["valid"]:.2f}, ' + \
           f'u={metrics["unique"]:.2f}, ' + \
           f'n={metrics["novel"]:.2f}, ' + \
           f's={metrics["score"]:.2f}, ' + \
           f'ms={metrics["m_stab"]:.2f}, ' + \
           f'as={metrics["a_stab"]:.2f}'

def conditional_fix(model, v, e, n, m, max_attempts=5):
    vv, ev = validate_sparsegs(v, e)
    for _ in range(max_attempts):
        vv, ev = model.sample_conditional(vv, ev, n, m)
        vv, ev = validate_sparsegs(vv, ev)
    return vv, ev

def sample_with_fix(model, num_samples, fix_type='remove'):
    v, e = model.sample(num_samples)
    if fix_type == 'remove':
        vv, ev = validate_sparsegs(v, e)
    elif fix_type == 'conditional':
        n = (v[..., 1] != -1).sum(dim=1) - 1
        m = (e[..., 2] != -1).sum(dim=1) - 1
        vv, ev = conditional_fix(model, v, e, n, m)
    else:
        raise Exception(f"Invalid fix_type {fix_type}")

    return vv.to(device='cpu', dtype=torch.int), ev.to(device='cpu', dtype=torch.int)

def resample_invalid_mols(model, num_samples, data_info, canonical=True, fix_type='remove', max_attempts=10):
    n = num_samples
    mols = []

    for _ in range(max_attempts):
        v, e = sample_with_fix(model, num_samples, fix_type=fix_type)
        vmols, _ = get_vmols(mols2smls(sparsegs2mols(v, e, data_info), canonical))
        mols.extend(vmols)
        n = num_samples - len(mols)
        if len(mols) == num_samples:
            break

    v_valid, e_valid = mols2sparsegs(mols, data_info)
    if n > 0:
        v_maybe, e_maybe = model.sample(n)
        return torch.cat((v_valid, v_maybe)), torch.cat((e_valid, e_maybe))
    else:
        return v_valid, e_valid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_metrics(gsmls, loaders, data_info):
    gmols = [Chem.MolFromSmiles(sml) for sml in gsmls]
    [Chem.Kekulize(mol) for mol in gmols]
    v, e = mols2sparsegs(gmols, data_info)

    metrics = evaluate_molecules(v, e, loaders, data_info, metrics_only=True)
    print(print_metrics(metrics))

if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    from utils.datasets import MOLECULAR_DATASETS, load_dataset
    data_info = MOLECULAR_DATASETS['qm9']
    loaders = load_dataset('qm9', 100, split=[0.8, 0.1, 0.1], order='bft')

    gsmls = [
            'CC1(C)CN1C(C)=O',
            'O=CC1=COC(=O)N=C1',
            'O=CC1(C=O)CN=CO1',
            'CCC1CC2C(O)C2O1',
            'CC1(C#N)C2CN=CN21',
            'CC1(C)OCC1CO',
            'O=C1C=CC2NC2CO1',
            'CC1C=CC(=O)C(C)N1',
            'COCCC1=CC=NN1',
            'CN1C(=O)C2C(O)C21C'
        ]
    test_metrics(gsmls, loaders, data_info)
