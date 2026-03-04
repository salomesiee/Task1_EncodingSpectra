import torch.nn.functional as F
import lightning as pl 
import torch 

from model import SpectraEncoder, UNet
from loss import CLIPLoss, ConstrastiveLoss


class CLIPLightningModel(pl.LightningModule):
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        self.model_ftir = UNet(in_channels, latent_dim)
        self.model_raman = UNet(in_channels, latent_dim)
        self.loss_fn = ConstrastiveLoss()

    def training_step(self, batch, batch_idx):
        xf, xr, labels = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xf, xr, labels = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding, labels)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-2, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return [optimizer], [scheduler]