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
        # # dm = QM9DM('data/qm9v6', 0, 256, tokenizer)
        # if args.dataset == 'QM9-jodo':
        #     dm = QM9DM(args.root, 2, 64, tokenizer, args)
        #     dataset_name = 'QM9'
        # elif args.dataset == 'GeomDrugs-JODO':
        #     dm = GeomDrugsJODODM(args.root, 2, 64, tokenizer, args)
        # else:
        #     raise NotImplementedError(f"dataset {args.dataset} not implemented")
        # smiles = load_smiles(path)
        # # mols = [Chem.MolFromSmiles(smi) for smi in smiles]
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

        # moses_out = dm.get_moses_metrics(mols)
        # print(moses_out)

        # properties = ['SNN', 'Frag', 'Scaf', 'FCD']
        # print(properties)
        # print('\t'.join([str(moses_out[prop]) for prop in properties]))

def load_selfies(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        selfies = [line.strip() for line in lines]
    return selfies
