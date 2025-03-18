import torch
from rdkit import Chem

from utils.molecular import BOND_DECODER, BOND_ENCODER, VALENCY_LIST, valency


def mol2g(mol, data_info):
    x = torch.full((data_info.max_atoms,), -1, dtype=torch.int8)
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        x[atom_idx] = data_info.atom_list.index(atom.GetAtomicNum())
    a = torch.full((data_info.max_atoms, data_info.max_atoms), -1, dtype=torch.int8)
    for bond in mol.GetBonds():
        c = BOND_ENCODER[bond.GetBondType()]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        a[i, j] = c
        a[j, i] = c
    return x, a

def unpad_g(x, a):
    atoms_exist = ((a!=-1).sum(dim=0) > 0) & (x != -1)
    atoms = x[atoms_exist]
    bonds = a[atoms_exist, :][:, atoms_exist]
    return atoms, bonds

def g2mol(x, a, data_info):
    mol = Chem.RWMol()

    atoms, bonds = unpad_g(x, a)

    for atom in atoms:
        mol.AddAtom(Chem.Atom(data_info.atom_list[atom]))

    for start, end in zip(*torch.nonzero(bonds!=-1, as_tuple=True)):
        if start > end:
            mol.AddBond(int(start), int(end), BOND_DECODER[bonds[start, end].item()])
            flag, valence = valency(mol)
            if flag:
                continue
            else:
                assert len(valence) == 2
                i = valence[0]
                v = valence[1]
                n = mol.GetAtomWithIdx(i).GetAtomicNum()
                if n in (7, 8, 16) and (v - VALENCY_LIST[n]) == 1:
                    mol.GetAtomWithIdx(i).SetFormalCharge(1)
    return mol

    
def permute_g(x, a, pi):
    px = x[pi]
    pa = a[pi, :]
    pa = pa[:, pi]
    return px, pa

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order #, depth_first_order, reverse_cuthill_mckee


def order_mol(mol, data_info, order='canonical'):
    n = mol.GetNumAtoms()
    match order:
        case 'canonical':
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
            pmol = Chem.MolFromSmiles(s)
            Chem.Kekulize(pmol)

        case 'bft':
            x, a = mol2g(mol, data_info)
            rand_pi = torch.cat((torch.randperm(n), torch.arange(n, data_info.max_atoms)))
            x, a = permute_g(x, a, rand_pi)
            pi = breadth_first_order(csr_matrix((a!=-1).to(torch.int8)), 0, directed=False, return_predecessors=False).tolist() + list(range((x!=-1).sum(), data_info.max_atoms))
            px, pa = permute_g(x, a, pi)
            pmol = g2mol(px, pa, data_info)
        
        case 'bft-c':
            x, a = mol2g(mol, data_info)
            pi = breadth_first_order(csr_matrix((a!=-1).to(torch.int8)), 0, directed=False, return_predecessors=False).tolist() + list(range((x!=-1).sum(), data_info.max_atoms))
            px, pa = permute_g(x, a, pi)
            pmol = g2mol(px, pa, data_info)

        case _:
            raise "Unknown order"
    return pmol

if __name__ == '__main__':
    from utils.datasets import MOLECULAR_DATASETS
    data_info = MOLECULAR_DATASETS['qm9']
    smls = [
            'CCC1(C)CN1C(C)=O',
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
    
    mols = [Chem.MolFromSmiles(s) for s in smls]
    [Chem.Kekulize(mol) for mol in mols]

    gs = [mol2g(mol, data_info) for mol in mols]
    mols_new = [g2mol(x, a, data_info) for (x, a) in gs]

    mols_mca = [order_mol(mol, data_info, order='canonical') for mol in mols]
    mols_bft = [order_mol(mol, data_info, order='bft') for mol in mols]

    from rdkit.Chem import Draw
    img = Draw.MolsMatrixToGridImage([mols, mols_new, mols_mca, mols_bft], subImgSize=(200, 200))
    img.save("molecules_g.png")


    

    

