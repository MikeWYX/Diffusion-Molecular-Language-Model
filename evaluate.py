import argparse
import torch
import pickle
from rdkit import Chem
from evaluation.eval_functions import get_2D_edm_metric
from tqdm import tqdm
from transformers import AutoTokenizer
from dataloader import MoleculeLoader
import selfies as sf

def eval_(path, mode):
    tokenizer = AutoTokenizer.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    if mode == 'selfies':
        dm = MoleculeLoader().my_load(task_name='zinc250k', splits=['train'])[0]
        selfies = load_selfies(path)
        mols = []
        for self in selfies:
            smi = sf.decoder(self)
            canonsmi = Chem.CanonSmiles(smi)
            mol = Chem.MolFromSmiles(canonsmi)
            mol = Chem.RemoveHs(mol)
            if mol is not None:
                mols.append(mol)
        edm2d_dout = get_2D_edm_metric(mols, dm)
        properties = ['atom_stable', 'mol_stable', 'Complete', 'Unique', 'Novelty']
        print(properties)
        print('\t'.join([str(edm2d_dout[prop]) for prop in properties]))
        print(edm2d_dout)


def load_selfies(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        selfies = [line.strip() for line in lines]
    return selfies
