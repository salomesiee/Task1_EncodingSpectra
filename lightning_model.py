import torch.nn.functional as F
import lightning
import torch 

from model import SpectraEncoder
from loss import CLIPLoss


class CLIPLightningModel(lightning.LightningModule):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.model_ftir = SpectraEncoder(in_channels, latent_dim)
        self.model_raman = SpectraEncoder(in_channels, latent_dim)
        self.loss_fn = CLIPLoss()

    def training_step(self, batch, batch_idx):
        xf, xr = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xf, xr = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)