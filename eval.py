import torch 
import pandas as pd 
import lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

from argparse import ArgumentParser
from pathlib import Path

from data_module import MultiModalDataModule
from lightning_model import CLIPLightningModel
from dataset import SpectralExtraction
import joblib
import numpy as np

parser = ArgumentParser()
parser = SpectralExtraction.add_specific_args(parser)
parser = CLIPLightningModel.add_specific_args(parser)
args = parser.parse_args()

model = CLIPLightningModel.load_from_checkpoint("runs/lightning_logs/version_7/checkpoints/epoch=11-step=924.ckpt")
dm = MultiModalDataModule(batch_size=128, args=args)
dm.setup("test")

logger=CSVLogger(save_dir="runs")
trainer = Trainer(devices=1, accelerator='auto', logger=logger)
preds_list = trainer.predict(model, dataloaders=dm.test_dataloader())

e_ftir = torch.cat([batch['ftir_embedding'] for batch in preds_list], dim=0).cpu().numpy()
e_raman = torch.cat([batch['raman_embedding'] for batch in preds_list], dim=0).cpu().numpy()
index_ftir = torch.cat([batch['index_ftir'] for batch in preds_list], dim=0).cpu().numpy()
index_raman = torch.cat([batch['index_raman'] for batch in preds_list], dim=0).cpu().numpy()

df_ftir = pd.DataFrame({
    'embedding': list(e_ftir),
    'index': list(index_ftir),
})
df_raman = pd.DataFrame({
    'embedding': list(e_raman),
    'index': list(index_raman),
})
df_out = pd.concat([df_ftir, df_raman], axis=0).drop_duplicates(subset=['index'])

df_all = pd.read_pickle(Path(args.cache_dir) / 'df_all_datasets_preprocessed.pkl')
df_all['index'] = df_all.index

df_preds = pd.merge(df_out, df_all, on='index', how='left')
df_preds.to_pickle(Path(logger.log_dir) / 'df_preds.pkl')
