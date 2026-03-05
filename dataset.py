from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch 

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd 
import numpy as np
import os 

from data_tools import PlasticDatabase
from vars import LABELS_MAPPING


class SpectralExtraction():
    def __init__(self, args, test_size=0.15, val_size=0.15):
        self.datasets_dir = Path(args.datasets_dir)
        self.cache_dir = Path(args.cache_dir)
        self.test_size = test_size
        self.val_size = val_size

        df_train, df_val, df_test = self.import_data()
        self.save_data(df_train, df_val, df_test)

    def import_data(self):
        ps = PlasticDatabase(load_local_libs=False, data_path=self.datasets_dir)
        df = ps.create_data(self.cache_dir / 'df_all.pkl')
        df['index'] = df.index

        strat_key = df['plastic'] + '_' + df['spectroscopy']

        df_train, df_temp = train_test_split(df, test_size=self.test_size + self.val_size, stratify=strat_key, random_state=42)
        strat_key_temp = df_temp['plastic'] + '_' + df_temp['spectroscopy']
        df_val, df_test = train_test_split(df_temp, test_size=self.test_size / (self.test_size + self.val_size), stratify=strat_key_temp, random_state=42)

        return df_train, df_val, df_test

    def save_data(self, df_train, df_val, df_test):
        # Split sets 
        df_train.to_pickle(self.cache_dir / 'df_train.pkl')
        df_val.to_pickle(self.cache_dir / 'df_val.pkl')
        df_test.to_pickle(self.cache_dir / 'df_test.pkl')

    @staticmethod
    def _add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--datasets_dir', type=str, required=True, help='Directory of the spectra datasets.')
        parser.add_argument('--cache_dir', type=str, default='datasets/cache', help='Directory of the formatted cached data.')
        parser.add_argument('--test_size', type=float, default=0.2, help='Test size fraction for train-test split.')
        return parser 

class MultiModalDataset(Dataset):
    def __init__(self, stage, transforms, args):
        self.stage = stage
        self.mapping = {c: i for i, c in LABELS_MAPPING.items()}  
        self.cache_dir = Path(args.cache_dir)
        self.transforms = transforms
        if stage=='train':
            data_path = self.cache_dir / 'df_train.pkl'
        elif stage == 'val': 
            data_path = self.cache_dir / 'df_val.pkl'
        elif stage == 'test': 
            data_path = self.cache_dir / 'df_test.pkl'
        if not os.path.exists(data_path):
            SpectralExtraction(args)
        df_init = pd.read_pickle(data_path)
        self.df = self.format_data(df_init) 

    def format_data(self, df):
        non_plastic_mask = (df['plastic'].isin(list(LABELS_MAPPING.values())))
        dfp = df[non_plastic_mask]

        df_raman = dfp[dfp['spectroscopy'] == 'raman'][['plastic', 'spectra', 'index']]
        df_ftir = dfp[dfp['spectroscopy'] == 'ftir'][['plastic', 'spectra', 'index']]

        df_pairs = df_raman.merge(df_ftir, on='plastic', suffixes=('_raman', '_ftir'))
        return df_pairs


    def __getitem__(self, index):
        ftir = self.df.iloc[index]['spectra_ftir']
        raman = self.df.iloc[index]['spectra_raman']
        
        plastic = self.df.iloc[index]['plastic']
        label = torch.tensor(self.mapping[plastic]) 
        index_ftir = self.df.iloc[index]['index_ftir']
        index_raman = self.df.iloc[index]['index_raman']

        if self.transforms:
            ftir = self.transforms(ftir)
            raman = self.transforms(raman)

        return ftir, raman, {'label': label, 'index_ftir': index_ftir, 'index_raman': index_raman}
    
    def __len__(self):
        return len(self.df)