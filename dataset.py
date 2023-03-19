from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

class CTRDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.df.iloc[idx, 0]
        y = np.array([y])

        x = self.df.iloc[idx, 1:]
        x = np.array([x])

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


