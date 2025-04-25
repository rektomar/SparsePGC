import json
import torch

from rdkit import RDLogger
from utils.datasets import BASE_DIR, load_dataset
from utils.train import train, evaluate
from utils.evaluate import count_parameters

from models import sparse_pgc, sparse_hpgc

MODELS = {
    **sparse_pgc.MODELS,
    **sparse_hpgc.MODELS,
}

BASE_DIR_TRN = f'{BASE_DIR}trn/'

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_printoptions(threshold=10_000, linewidth=200)
    RDLogger.DisableLog('rdApp.*')

    dataset = 'zinc250k'

    backends = [
        'spgc_rtree'
    ]

    for backend in backends:
        with open(f'config/{dataset}/{backend}.json', 'r') as f:
            hyperpars = json.load(f)

        loaders = load_dataset(dataset, hyperpars['batch_size'], [0.8, 0.1, 0.1], order=hyperpars['order'])

        model = MODELS[hyperpars['model']](dataset, hyperpars['model_hpars'])
        print(dataset)
        print(json.dumps(hyperpars, indent=4))
        print(model)
        print(f'The number of parameters is {count_parameters(model)}.')
        print(hyperpars['order'])

        train(model, loaders, hyperpars, BASE_DIR_TRN)
        metrics = evaluate(loaders, hyperpars, BASE_DIR_TRN, compute_nll=True)

        print("\n".join(f'{key:<16}{value:>10.4f}' for key, value in metrics.items()))
