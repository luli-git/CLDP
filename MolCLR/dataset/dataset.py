# import os
import csv

# import math
# import time
# import random
# import networkx as nx
import numpy as np

# from copy import deepcopy

import torch
import torch.nn.functional as F

# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import Subset
# import torchvision.transforms as transforms

# from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

# import rdkit
from rdkit import Chem

# from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT

# from rdkit.Chem import AllChem
from torch_geometric.data import Batch
import pandas as pd
from torch.utils.data import Dataset as TorchDataset

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_csv_path):
        super(MoleculeDataset, self).__init__()

        merged_df = pd.read_csv(data_csv_path)

        # Store data and initialize other parameters
        self.texts = merged_df["description"].tolist()
        self.smiles_data = merged_df["Isomeric_SMILES"].tolist()

    def get(self, index):
        text = self.texts[index]
        positive_molecule = self.smiles_data[index]

        # for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(positive_molecule)
            if mol is None:  # Check if the molecule creation was successful
                print(f"Could not parse SMILES string: {positive_molecule}")

            # The following code assumes there is existing logic to convert a mol object to a graph representation
            # This logic will be represented here as `mol_to_graph` which needs to be implemented
            N = mol.GetNumAtoms()

            type_idx = []
            chirality_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

            x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
            x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
            x = torch.cat([x1, x2], dim=-1)

            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_feat.append(
                    [
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                    ]
                )
                edge_feat.append(
                    [
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                    ]
                )

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

            # Construct a Data object for the original molecule
            original_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        except Exception as e:
            print(f"Error processing SMILES string: {positive_molecule}. Error: {e}")
        return {"graph": original_data, "text": text}

    def __len__(self):
        return len(self.smiles_data)

    def len(self):
        return self.__len__()


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def __len__(self):
        return self.batch_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_csv_path=self.data_path)
        train_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader

    def collate_fn(self, batch):
        graphs = [item["graph"] for item in batch]
        texts = [item["text"] for item in batch]

        # Batch graphs using PyTorch Geometric's Batch class
        batched_graphs = Batch.from_data_list(graphs)

        return {"graphs": batched_graphs, "texts": texts}

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices, indices[:split]

        # define samplers for obtaining training and validation batches

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return train_loader


class DrugDataset(TorchDataset):
    def __init__(self, data_csv_path):
        # Load data
        merged_df = pd.read_csv(data_csv_path)

        # Store data and initialize other parameters
        self.texts = merged_df["description"].tolist()
        self.molecule_representations = merged_df["Isomeric_SMILES"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        positive_molecule = self.molecule_representations[idx]

        return text, positive_molecule
