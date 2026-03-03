from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd 
import os 

from data_tools import PlasticDatabase

class SpectralExtraction():
    def __init__(self, args, test_size=0.2):
        self.datasets_dir = Path(args.datasets_dir)
        self.cache_dir = Path(args.cache_dir)
        self.test_size = test_size

        dfi = self.import_data()
        print("df", dfi.head())

        df = self.format_data(dfi)
        self.save_data(df)

    def import_data(self):
        ps = PlasticDatabase(load_local_libs=False, data_path=self.datasets_dir)
        df = ps.create_data(self.cache_dir / 'df_all.pkl')
        return df

    def format_data(self, df):
        dfp = df[df['plastic'] != 'non_plastic']
        # Divide Raman from FTIR
        df_raman = dfp[dfp['spectroscopy'] == 'raman'][['plastic', 'spectra']]
        df_ftir = dfp[dfp['spectroscopy'] == 'ftir'][['plastic', 'spectra']]

        # Dataframe of pairs
        df_pairs = df_raman.merge(df_ftir, on='plastic', suffixes=('_raman', '_ftir'))
        return df_pairs

    def save_data(self, df):
        # Split sets 
        df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=42)
        df_train.to_pickle(self.cache_dir / 'df_trainval.pkl')
        df_test.to_pickle(self.cache_dir / 'df_test.pkl')

    @staticmethod
    def _add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--datasets_dir', type=str, required=True, help='Directory of the spectra datasets.')
        parser.add_argument('--cache_dir', type=str, default='datasets/cache', help='Directory of the formatted cached data.')
        parser.add_argument('--test_size', type=float, default=0.2, help='Test size fraction for train-test split.')
        return parser 

class MultiModalDataset(Dataset):
    def __init__(self, train, transforms, args):
        self.cache_dir = Path(args.cache_dir)
        self.transforms = transforms
        if train:
            data_path = self.cache_dir / 'df_trainval.pkl'
        else: 
            data_path = self.cache_dir / 'df_test.pkl'
        if not os.path.exists(data_path):
            SpectralExtraction(args)
        self.df = pd.read_pickle(data_path)

    def __getitem__(self, index):
        ftir = self.df.iloc[index]['spectra_ftir']
        raman = self.df.iloc[index]['spectra_raman']
        if self.transforms:
            ftir = self.transforms(ftir)
            raman = self.transforms(raman)
        return ftir, raman 
    
    def __len__(self):
        return len(self.df)