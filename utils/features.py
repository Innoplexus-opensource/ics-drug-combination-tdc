# Graph based featurizer adapted from https://github.com/bowang-lab/CongFu/tree/main

import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from config import drug_representation_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

drug_model = AutoModel.from_pretrained(drug_representation_model, deterministic_eval=True, trust_remote_code=True)
drug_tokenizer = AutoTokenizer.from_pretrained(drug_representation_model, trust_remote_code=True)

def _get_drug_tokens(smiles: str) -> Data:

    '''Converts SMILES to PyG Data format'''

    smiles = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_mol_dict(df: pd.DataFrame) -> dict:

    '''Returns the mapping between IDs and molecular graphs'''

    mols = pd.concat([
        df.rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'})[['id', 'drug']],
        df.rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})[['id', 'drug']]
    ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=['id'])

    dct = {}
    for _, x in tqdm(mols.iterrows(), total=len(mols)):
        dct[x['id']] = _get_drug_tokens(x['drug'])
    return dct


def _get_drug_embedding(smiles: str) -> Data:

    '''Converts SMILES to embeddings'''

    smiles = Chem.CanonSmiles(smiles) 
    smiles = [smiles]
    inputs = drug_tokenizer(smiles, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = drug_model(**inputs)
    return outputs.pooler_output[0]

def get_mol_embed_dict(df: pd.DataFrame) -> dict:

    '''Returns the mapping between IDs and molecular embeddings'''

    mols = pd.concat([
        df.rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'})[['id', 'drug']],
        df.rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})[['id', 'drug']]
    ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=['id'])

    dct = {}
    for _, x in tqdm(mols.iterrows(), total=len(mols)):
        dct[x['id']] = _get_drug_embedding(x['drug'])
    return dct