

class FTIRandRamanSpectra():
    def __init__(self, df):
        self.df = df

    def build_bimodal_db(self, df):
        # Divide Raman from FTIR
        df_raman = df[df['spectroscopy'] == 'Raman'][['plastic', 'spectra']]
        df_ftir = df[df['spectroscopy'] == 'FTIR'][['plastic', 'spectra']]

        # Dataframe of pairs
        pairs = df_raman.merge(df_ftir, on='plastic', suffixes=('_dr', '_df'))

        # Extract pairs 
        pairs_list = list(zip(pairs['spectra_dr'], pairs['spectra_df']))   

        # Preprocess them 