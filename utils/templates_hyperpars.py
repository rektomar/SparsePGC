import os
import json

from utils.datasets import MOLECULAR_DATASETS

NUM_EPOCHS = 40


def template_shpgc_btree(nl: int=4,            ns: int=40, ni: int=40): return {"nl": nl,           "ns": ns, "ni": ni}
def template_shpgc_rtree(nl: int=4, nr: int=1, ns: int=40, ni: int=40): return {"nl": nl, "nr": nr, "ns": ns, "ni": ni}
def template_shpgc_ptree(nl: int=4,            ns: int=40, ni: int=40): return {"nl": nl,           "ns": ns, "ni": ni}



def template_sort(
    dataset: str,
    model: str,
    order: str = "canonical",
    nc: int = 100,
    nr: int = None,
    backend: str = "btree",
    bvt_hpars: dict = template_shpgc_btree(),
    be_hpars: dict = template_shpgc_btree(),
    bet_hpars: dict = template_shpgc_btree(),
    batch_size: int = 1000,
    lr: float = 0.05,
    seed: int = 0
):
    hpars = {
        "dataset": dataset,
        "order": order,
        "model": model,
        "model_hpars": {
            "nc": nc,
            "backend": backend,
            "bvt_hpars": bvt_hpars,
            "be_hpars": be_hpars,
            "bet_hpars": bet_hpars,
            "device": "cuda"
        },
        "optimizer": "adam",
        "optimizer_hpars": {
            "lr": lr,
            "betas": [
                0.9,
                0.82
            ]
        },
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "seed": seed
    }

    if nr is not None:
        hpars["model_hpars"]["nr"] = nr

    return hpars


HYPERPARS_TEMPLATES = [
    template_sort,
]


if __name__ == '__main__':
    for dataset in ['qm9', 'zinc250k']:
        dir = f'config/{dataset}'
        # if os.path.isdir(dir) != True:
        #     os.makedirs(dir)
        for template in HYPERPARS_TEMPLATES:
            hyperpars = template(dataset, 'sparse_hpgc')
            print(json.dumps(hyperpars, indent=4))
