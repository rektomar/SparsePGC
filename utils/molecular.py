import re
import torch

from rdkit import Chem


# VALENCY_LIST has to change for different datasets.
VALENCY_LIST = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

BOND_ENCODER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
BOND_DECODER = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}


def mol2sparseg(mol, atom_list):
    atom_tensor = torch.zeros(mol.GetNumAtoms(), 2, dtype=torch.int8)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_tensor[i, 0] = atom.GetIdx()
        atom_tensor[i, 1] = atom_list.index(atom.GetAtomicNum())

    bond_tensor = torch.zeros(mol.GetNumBonds(), 3, dtype=torch.int8)
    for i, bond in enumerate(mol.GetBonds()):
        bond_tensor[i, 0] = bond.GetBeginAtomIdx()
        bond_tensor[i, 1] = bond.GetEndAtomIdx()
        bond_tensor[i, 2] = BOND_ENCODER[bond.GetBondType()]

    return atom_tensor, bond_tensor

def sparseg2mol(atom_tensor, bond_tensor, atom_list):
    mol = Chem.RWMol()

    for _, atom in atom_tensor:
        mol.AddAtom(Chem.Atom(atom_list[int(atom)]))

    for start, end, bond_type in bond_tensor:
        mol.AddBond(int(start), int(end), BOND_DECODER[int(bond_type)])
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

def sparsegs2mols(atom_tensor, bond_tensor, atom_list):
    return [sparseg2mol(v, e, atom_list) for v, e in zip(atom_tensor, bond_tensor)]

def mols2smls(mols, canonical=True):
    return [Chem.MolToSmiles(mol, canonical=canonical) for mol in mols]

def valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as error:
        valence = list(map(int, re.findall(r'\d+', str(error))))
        return False, valence

def correct(mol):
    while True:
        flag, atomid_valence = valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            queue = []
            for b in mol.GetAtomWithIdx(atomid_valence[0]).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                bond_index = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if bond_index > 0:
                    mol.AddBond(start, end, BOND_DECODER[bond_index])

    return mol

def correct_mols(atom_tensor, bond_tensor, atom_list):
    return [correct(mol) for mol in sparsegs2mols(atom_tensor, bond_tensor, atom_list)]

def get_valid(sml):
    mol = Chem.MolFromSmiles(sml)
    if mol is not None and '.' not in sml:
        Chem.Kekulize(mol)
        return mol, sml
    else:
        return None

def get_vmols(smls):
    vmols = []
    vsmls = []
    for s in smls:
        v = get_valid(s)
        if v is not None:
            vmols.append(v[0])
            vsmls.append(v[1])

    return vmols, vsmls

def isvalid(mol, canonical=True):
    sml = Chem.MolToSmiles(mol, canonical=canonical)
    if Chem.MolFromSmiles(sml) is not None and mol is not None and '.' not in sml:
        return True
    else:
        return False


if __name__ == '__main__':
    # 10 samples from the QM9 dataset
    max_atom = 9
    atom_list = [0, 6, 7, 8, 9]
    smiles = [
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

    for sa in smiles:
        mol = Chem.MolFromSmiles(sa)
        Chem.Kekulize(mol)
        v, e = mol2sparseg(mol, atom_list)
        #print(v, e)
        mol = sparseg2mol(v, e, atom_list)
        sb = Chem.MolToSmiles(mol, kekuleSmiles=True)
        print(f'{sa}  {sb}')
