from lightning import Trainer

from data_module import MultiModalDataModule
from lightning_model import CLIPLightningModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__": 
    # Get Plastic data 
    dm = MultiModalDataModule(data_dir='datasets/cache', batch_size=256)
    model = CLIPLightningModel(in_channels=1, latent_dim=100)

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(max_epochs=10, accelerator='auto', callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model, dm)