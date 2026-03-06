import torch.nn.functional as F
import lightning as pl 
import torch 
from argparse import ArgumentParser

from model import SpectraEncoder, UNet
from loss import CLIPLoss, ConstrastiveLoss, SimilarityLoss, SymmetricKL, ConstrastiveIdeaLoss


class CLIPLightningModel(pl.LightningModule):
    def __init__(self, in_channels, latent_dim, model, loss, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.get_model(model)
        self.loss_fn = self.get_loss_fn(loss)

        self.model_ftir = self.model(in_channels, latent_dim)
        self.model_raman = self.model(in_channels, latent_dim)

    def get_model(self, model_name):
        if model_name == 'unet':
            return UNet
        elif model_name == 'encoder':
            return SpectraEncoder
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def get_loss_fn(self, loss_name):
        if loss_name == 'clip':
            return CLIPLoss()
        elif loss_name == 'similarity':
            return SimilarityLoss()
        elif loss_name == 'kl':
            return SymmetricKL()
        elif loss_name == 'idea':
            return ConstrastiveIdeaLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def training_step(self, batch, batch_idx):
        xf, xr, info = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding, info['label'])
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xf, xr, info = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding, info['label'])
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        xf, xr, info = batch 
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)

        loss = self.loss_fn(xf_encoding, xr_encoding, info['label'])
        self.log("test_loss", loss, sync_dist=True)
        return loss 

    def predict_step(self, batch, batch_idx):
        xf, xr, info = batch 
        index_ftir, index_raman = info['index_ftir'], info['index_raman']
        xf_encoding = F.normalize(self.model_ftir(xf), dim=-1)
        xr_encoding = F.normalize(self.model_raman(xr), dim=-1)
        return {'ftir_embedding': xf_encoding, 'raman_embedding': xr_encoding, 'index_ftir': index_ftir, 'index_raman': index_raman}

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model", default='unet', type=str, help="Model architecture.")
        parser.add_argument("--loss", default='kl', type=str, help="Loss function.")
        parser.add_argument("--in_channels", default=1, type=int, help="In channels.")
        parser.add_argument("--latent_dim", default=128, type=int, help="Embedding output size.")
        return parser
