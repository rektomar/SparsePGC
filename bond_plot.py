import torch

from utils.datasets import load_dataset
from pylatex import Document, TikZ, Axis, NoEscape

DATASET_NAMES = {'qm9': 'QM9', 'zinc250k': 'Zinc250k'}

def create_matrix(n, m):
    n_max, m_max = int(n.max().item())+1, int(m.max().item())+1
    matrix = torch.zeros((n_max, m_max))
    for (n_i, m_i) in zip(n, m):
        matrix[n_i, m_i] += 1
    matrix = matrix / matrix.sum(1, keepdim=True)
    matrix[torch.isnan(matrix)] = 0
    return matrix

def fill_coords(matrix):
    s = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            s.append(f'({i},{j}) [{matrix[i, j]}]')
        s.append('\n\n')
    s = ' '.join(s)
    return s


def create_document(dataset, matrix):
    max_atoms, max_bonds = matrix.shape

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\pgfplotsset{compat=1.18}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{colorbrewer}'))

    with doc.create(TikZ(options=NoEscape(r'font=\footnotesize'))) as tikz:
        axis_options = NoEscape('colormap/YlOrBr, colorbar, axis on top, xlabel=Num. Atoms, ylabel= Num. Bonds,' +
                        f'title={DATASET_NAMES[dataset]},'
                        r'xmin=-0.5,' +
                        r'ymin=-0.5,' +
                        f'xmax={max_atoms-0.5},' +
                        f'ymax={max_bonds-0.5},')
        with tikz.create(Axis(options=axis_options)) as ax:
            s = fill_coords(matrix)
            ax.append(NoEscape(f'\\addplot[matrix plot*, point meta=explicit] coordinates {{\n {s} \n}};'))

    doc.generate_pdf('plots/bond_plot', clean_tex=False)


if __name__ == "__main__":
    dataset = 'zinc250k'

    loader = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='bft')
    m = torch.tensor([b['m'] for b in loader['loader_trn'].dataset], dtype=torch.int8)
    n = torch.tensor([b['n'] for b in loader['loader_trn'].dataset], dtype=torch.int8)

    matrix = create_matrix(n, m)
    create_document(dataset, matrix)
    
