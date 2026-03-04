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
            self.dtrain= MultiModalDataset(stage='train', transforms=self.transforms, args=self.args)
            self.dval = MultiModalDataset(stage='val', transforms=self.transforms, args=self.args)

        if stage == "test":
            self.dtest = MultiModalDataset(stage='test', transforms=self.transforms, args=self.args)

    def train_dataloader(self):
        return DataLoader(self.dtrain, batch_size=self.batch_size, shuffle=True, num_workers=17)
    
    def val_dataloader(self):
        return DataLoader(self.dval, batch_size=self.batch_size, shuffle=False, num_workers=17)
    
    def test_dataloader(self):
        return DataLoader(self.dtest, batch_size=self.batch_size, shuffle=False, num_workers=17)

