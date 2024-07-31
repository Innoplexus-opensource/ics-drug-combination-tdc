import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl


DRUG1_ID_COLUMN_NAME = "Drug1_ID"
DRUG2_ID_COLUMN_NAME= "Drug2_ID"
CELL_LINE_COLUMN_NAME = "Cell_Line_ID"


def get_datasets(train, validation, test):
    
    '''Prepares all input datasets for the model based on the evaluation setup and synergy score'''

    dataset = pd.concat((train, test, validation))
    cell_lines = dataset[["Cell_Line_ID", "CellLine"]].drop_duplicates("Cell_Line_ID").set_index("Cell_Line_ID").to_dict()["CellLine"]
    cell_lines = {k: np.concatenate(v) for k, v in cell_lines.items()}
    cell_lines = pd.DataFrame(cell_lines).T
    cell_lines = cell_lines.astype(np.float32)
    train_dataset = train[['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Drug1', 'Drug2', 'Y']]
    val_dataset = validation[['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Drug1', 'Drug2','Y']]
    test_dataset = test[['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Drug1', 'Drug2', 'Y']]

    return dataset, train_dataset, val_dataset, test_dataset, cell_lines


class DrugCombDataset(Dataset):

    '''Dataset class that returns all model inputs and a target'''

    def __init__(self, drugcomb, cell_lines, mol_mapping, transform=None, inference=False):
        self.drugcomb = drugcomb
        self.mol_mapping = mol_mapping
        self.cell_lines = cell_lines
        self.inference = inference
        if inference:
            self.targets = torch.zeros((drugcomb.shape[0],))
        else:
            self.targets = torch.from_numpy(drugcomb['Y'].values)
        self.transform = transform
    def __len__(self):
        return len(self.drugcomb)

    def __getitem__(self, idx):
        sample = self.drugcomb.iloc[idx]

        drug1 = sample[DRUG1_ID_COLUMN_NAME]
        drug2 = sample[DRUG2_ID_COLUMN_NAME]
        drug1_tokens = self.mol_mapping[drug1]
        drug2_tokens = self.mol_mapping[drug2]

        if self.transform:
            drug1_tokens = self.transform(drug1_tokens)
            drug2_tokens = self.transform(drug2_tokens)

        cell_line_name = sample[CELL_LINE_COLUMN_NAME]
        cell_line_embeddings = self.cell_lines.loc[cell_line_name].values.flatten()
        cell_line_embeddings = torch.tensor(cell_line_embeddings)

        target = self.targets[idx].unsqueeze(-1).float()
        
        return (drug1_tokens, drug2_tokens, cell_line_embeddings, target)


class DataModule(pl.LightningDataModule):
    '''
    Defines pytorch lightning data module for DrugComb dataset
    '''
    
    def __init__(self,
                 train, validation, test, drug_featurizer,
                 batch_size=16, device="cpu"):
        super().__init__()
        
        dataset, self.train_dataset, self.val_dataset, self.test_dataset, self.cell_lines = get_datasets(train, validation, test)
        self.mol_mapping = drug_featurizer(dataset)
        
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_set = DrugCombDataset(self.train_dataset, self.cell_lines, self.mol_mapping)
        self.val_set = DrugCombDataset(self.val_dataset, self.cell_lines, self.mol_mapping)
        self.test_set = DrugCombDataset(self.test_dataset, self.cell_lines, self.mol_mapping)
        
    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size = self.batch_size, num_workers=1, shuffle=True, prefetch_factor=10, pin_memory=True, pin_memory_device='cuda')

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size= self.batch_size, num_workers=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size= self.batch_size, num_workers=1, shuffle=False)