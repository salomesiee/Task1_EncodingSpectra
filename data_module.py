from torch.utils.data import DataLoader
import lightning
import torch 

from dataset import FTIRDataset, MultiModalDataset


class FTIRDataModule(lightning.LightningDataModule):
    def __init__(self, data, labels, batch_size, val_split=0.2):
        super().__init__()
        self.dataset = FTIRDataset(data, labels)
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage):
        n_val = int(len(self.dataset) * self.val_split)
        n_train = len(self.dataset) - n_val
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


class MultiModalDataModule(lightning.LightningDataModule):
    def __init__(self, data_raman, data_ftir, batch_size, val_split=0.2):
        super().__init__()
        self.dataset = MultiModalDataset(data_raman, data_ftir,)
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage):
        n_val = int(len(self.dataset) * self.val_split)
        n_train = len(self.dataset) - n_val
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
