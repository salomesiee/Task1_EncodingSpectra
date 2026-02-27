import pandas as pd

import torch
from lightning import Trainer
from data_tools import PlasticDatabase

from data_module import MultiModalDataModule
from lightning_model import CLIPLightningModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__": 
    # Get Plastic data 
    df = PlasticDatabase(
        data_path='/Users/salomepapereux/PhD/datasets', 
        load_local_libs=False
    ).create_data("datasets/cache/plastic_database.pkl")

    df = pd.read_pickle("datasets/cache/plastic_database.pkl")
    df.to_csv("datasets/cache/to_del.csv")

    data_ftir = torch.randn(200, 1, 4000)
    data_raman = torch.randn(200, 1, 4000)

    labels = torch.randn(200)
    batch_size = 32

    dm = MultiModalDataModule(data_ftir, data_raman, batch_size)
    model = CLIPLightningModel(in_channels=1, latent_dim=100)

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = Trainer(max_epochs=10, accelerator='auto', callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model, dm)