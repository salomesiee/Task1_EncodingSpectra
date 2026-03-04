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
        df_train, df_val, df_test = self.format_data(df_train, df_val, df_test)
        self.save_data(df_train, df_val, df_test)

    def import_data(self):
        ps = PlasticDatabase(load_local_libs=False, data_path=self.datasets_dir)
        df = ps.create_data(self.cache_dir / 'df_all.pkl')

        train_end = int((1 - self.test_size - self.val_size) * len(df)) 
        val_end   = int((1 - self.test_size) * len(df))
        train, val, test = np.split(df.sample(frac=1, random_state=42), [train_end, val_end])
        df_train = pd.DataFrame(train, columns=df.columns)
        df_test = pd.DataFrame(test, columns=df.columns)
        df_val = pd.DataFrame(val, columns=df.columns)

        return df_train, df_val, df_test

    def format_data(self, *dfs):
        formatted_dfs = []
        for df in dfs:
            non_plastic_mask = (df['plastic'].isin(list(LABELS_MAPPING.values())))
            dfp = df[non_plastic_mask]
            # Divide Raman from FTIR
            df_raman = dfp[dfp['spectroscopy'] == 'raman'][['plastic', 'spectra']]
            df_ftir = dfp[dfp['spectroscopy'] == 'ftir'][['plastic', 'spectra']]

            # Dataframe of pairs
            df_pairs = df_raman.merge(df_ftir, on='plastic', suffixes=('_raman', '_ftir'))
            formatted_dfs.append(df_pairs)
        return formatted_dfs

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
        self.df = pd.read_pickle(data_path)

    def __getitem__(self, index):
        ftir = self.df.iloc[index]['spectra_ftir']
        raman = self.df.iloc[index]['spectra_raman']
        
        plastic = self.df.iloc[index]['plastic']
        label = torch.tensor(self.mapping[plastic]) 

        if self.transforms:
            ftir = self.transforms(ftir)
            raman = self.transforms(raman)
        return ftir, raman, label
    
    def __len__(self):
        return len(self.df)