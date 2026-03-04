from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser

from data_module import MultiModalDataModule
from dataset import SpectralExtraction
from lightning_model import CLIPLightningModel

parser = ArgumentParser()
parser = SpectralExtraction._add_specific_args(parser)
args = parser.parse_args()

dm = MultiModalDataModule(batch_size=128, args=args)
model = CLIPLightningModel(in_channels=1)

checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
earlystop_cb = EarlyStopping(monitor="val_loss", patience=15, mode="min")

trainer = Trainer(max_epochs=100, accelerator='auto', callbacks=[checkpoint_cb, earlystop_cb])
trainer.fit(model, dm)