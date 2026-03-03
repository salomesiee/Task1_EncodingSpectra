from torch.utils.data import DataLoader, random_split
import lightning
import torch 

from dataset import MultiModalDataset
from transforms import TransformsComposer


class MultiModalDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size, args, val_split=0.2):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.val_split = val_split
        self.transforms = TransformsComposer(
            preprocessing_funcs=['baseline', 'smoothing', 'interpolate'],
            normalize=True,
        )

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.dtrainval = MultiModalDataset( train=True, transforms=self.transforms, args=self.args)
            n_val = int(self.val_split * len(self.dtrainval))
            n_train = len(self.dtrainval) - n_val
            self.dtrain, self.dval = random_split(self.dtrainval, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        if stage == "test":
            self.dtest = MultiModalDataset(train=False, transforms=self.transforms, args=self.args)

    def train_dataloader(self):
        return DataLoader(self.dtrain, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dval, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dtest, batch_size=self.batch_size, shuffle=False)

