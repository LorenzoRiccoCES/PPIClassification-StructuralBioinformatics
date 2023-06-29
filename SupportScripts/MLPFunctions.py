from torch import nn
import torch
from torch.nn import Conv2d, MaxPool2d, Linear
from torchinfo import summary
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Lambda
from sklearn.preprocessing import LabelEncoder
import pandas as pd
    

def encode_data(df):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])
    return df_encoded



class ProteinDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert data to numpy array and then to tensor
        x = torch.tensor(self.X.iloc[idx])

        return x
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        return out