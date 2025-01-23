import datasets
import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
import json


class DiffusionLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('/NAS/luoyc/wuyux/data/zinc250k', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        removed_columns = ['index', 'smiles', 'selfies', 'logP', 'qed', 'SAS']
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns=removed_columns)
        return dataset

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['selfies'], max_length=128, truncation=True, add_special_tokens=False)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings

class MoleculeLoader:
    def __init__(self):
        pass

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('/NAS/luoyc/wuyux/data/zinc250k', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        mols = []
        for smi in dataset["smiles"]:
            canonsmi = Chem.CanonSmiles(smi)
            mol = Chem.MolFromSmiles(canonsmi)
            mol = Chem.RemoveHs(mol)
            if mol is not None:
                mols.append(mol)
        return mols

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

class Zinc250kLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset('/NAS/luoyc/wuyux/data/zinc250k', split='train')
        
    def _get_splits(self):        
        # Load test indices
        selfies_list = self.dataset['selfies']
        with open('/NAS/luoyc/wuyux/data/zinc250k/valid_idx_zinc250k.json') as f:
            test_idx = np.array(json.load(f))    
        # Calculate train indices
        train_idx = np.array(list(set(np.arange(len(selfies_list))).difference(set(test_idx))))
       
        return {
            'train': train_idx.tolist(),
            'test': test_idx.tolist()
        }

    def _load_train(self):
        # Get split indices
        splits = self._get_splits()
        split_idx = splits['train']
        # Select subset for this split
        dataset = self.dataset.select(split_idx)
        
        print(f'Example in {split} set:')
        print(dataset[0])
        
        # Process features
        removed_columns = ['index', 'smiles', 'selfies', 'logP', 'qed', 'SAS']
        dataset = dataset.map(
            partial(self.convert_to_features, tokenizer=self.tokenizer), 
            batched=True, 
            remove_columns=removed_columns
        )
        return dataset

    def _load_test(self):
        splits = self._get_splits()
        split_idx = splits['test']
        dataset = self.dataset.select(split_idx)
        return dataset

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        """Convert examples to features"""
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['selfies'],
            max_length=128,
            truncation=True,
            add_special_tokens=False
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

class NPLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def _get_splits(self):        
        # Load test indices
        selfies_list = self.dataset['selfies']
        with open('/NAS/luoyc/wuyux/data/zinc250k/valid_idx_zinc250k.json') as f:
            test_idx = np.array(json.load(f))    
        # Calculate train indices
        train_idx = np.array(list(set(np.arange(len(selfies_list))).difference(set(test_idx))))
       
        return {
            'train': train_idx.tolist(),
            'test': test_idx.tolist()
        }

    def _load_train(self):
        # Get split indices
        splits = self._get_splits()
        split_idx = splits['train']
        # Select subset for this split
        dataset = self.dataset.select(split_idx)
        
        print(f'Example in {split} set:')
        print(dataset[0])
        
        # Process features
        removed_columns = ['index', 'smiles', 'selfies', 'logP', 'qed', 'SAS']
        dataset = dataset.map(
            partial(self.convert_to_features, tokenizer=self.tokenizer), 
            batched=True, 
            remove_columns=removed_columns
        )
        return dataset

    def _load_test(self):
        splits = self._get_splits()
        split_idx = splits['test']
        dataset = self.dataset.select(split_idx)
        return dataset

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        """Convert examples to features"""
        input_encodings = tokenizer.batch_encode_plus(
            example_batch['selfies'],
            max_length=128,
            truncation=True,
            add_special_tokens=False
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }