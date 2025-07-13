# Sparse Probabilistic Graph Circuits (SparsePGCs)



## 1. Install

Clone this repository.
```
git clone git@github.com:rektomar/SparsePGCs.git
```

Go to the SparsePGCs directory.
```
cd SparsePGCs
```

Set up the environment.
```
conda create --name spgc python=3.10

source activate spgc

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install rdkit==2024.3.6
pip install tqdm==4.67.0
pip install pandas==2.2.3
pip install pylatex==1.4.2
pip install scipy==1.14.1
pip install fcd_torch==1.0.7
pip install scikit-learn==1.6.0
pip install git+https://github.com/fabriziocosta/EDeN.git
```

## 2. Preprocess
The following command will download and preprocess two versions of the QM9 dataset. `qm9_sort.pt` contains molecules with the canonical ordering of the atoms. `qm9_perm.pt` contains molecules with a random ordering of the atoms.
```
python -m utils.datasets
```
## 3. Train
`config/qm9/` contains JSON files with the MolSPN variants' hyper-parameters. Change the hyper-parameters based on your preferences and then run the following command.
```
python -m train
```
It will train all the MolSPN variants (or only the selected ones if you change the list of `names` in `train.py`).

The resulting models will be stored in `results/training/model_checkpoint/`, and the corresponding illustrations of unconditional molecule generation, along with the metrics assessing the performance of the models, will be stored in `results/training/model_evaluation/`.

<img src="plots/unconditional_generation.png" width="500"/>

*Unconditional samples of molecular graphs from the sort variant of MolSPNs (`molspn_zero_sort`).*

## 4. Gridsearch
`gridsearch_hyperpars.py` contains hyper-parameter grids to find a suitable architecture for the MolSPN variants. Change the hyper-parameter grids based on your preferences, and then run the following command.
```
nohup python -m gridsearch > gridsearch.log &
```
This command will run the script in the background, submitting jobs to your SLURM cluster. The resulting models, metrics, and output logs will be stored in `results/gridsearch/model_checkpoint/`, `results/gridsearch/model_evaluation/`, and `results/gridsearch/model_outputlogs/`, respectively.

To reproduce the results in the paper (Table 1), keep the current settings in `gridsearch_hyperpars.py`. Then, after completing all the SLURM jobs, run the following command.
```
python -m gridsearch_evaluate
```
It will produce Table 1 from the paper (both in the `.pdf` and `.tex` formats).

## 4. Conditional Molecule Generation
Run the following command to generate new molecules conditionally on a known molecule.
```
python -m conditional_sampling
```
To impose a known structure of the generated molecules, change `patt_smls` in `conditional_sampling.py`. Similarly, to select a model from which to generate the samples, change `model_path`.

<img src="plots/conditional_generation.png" width="500"/>

*Conditional samples of molecular graphs from the sort variant of MolSPNs (`molspn_zero_sort`). The known part of a molecule is highlighted in blue.*
