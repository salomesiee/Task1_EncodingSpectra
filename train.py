from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from argparse import ArgumentParser

from data_module import MultiModalDataModule
from dataset import SpectralExtraction
from lightning_model import CLIPLightningModel

parser = ArgumentParser()
parser = SpectralExtraction.add_specific_args(parser)
parser = CLIPLightningModel.add_specific_args(parser)
args = parser.parse_args()

dm = MultiModalDataModule(batch_size=256, args=args)
model = CLIPLightningModel(**vars(args))

checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
earlystop_cb = EarlyStopping(monitor="val_loss", patience=15, mode="min")


logger = CSVLogger(save_dir="runs")
trainer = Trainer(max_epochs=200, accelerator='auto', callbacks=[checkpoint_cb, earlystop_cb], logger=logger)
trainer.fit(model, dm)