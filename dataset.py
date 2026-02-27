from torch.utils.data import Dataset, DataLoader


class FTIRDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels 

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


class MultiModalDataset(Dataset):
    def __init__(self, data_ftir, data_raman):
        self.data_ftir = data_ftir 
        self.data_raman = data_raman 

    def __getitem__(self, index):
        return self.data_ftir[index], self.data_raman[index]
    
    def __len__(self):
        return len(self.data_ftir)