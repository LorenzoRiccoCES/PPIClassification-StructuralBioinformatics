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
    # Create a copy of the input DataFrame to avoid modifying the original data
    df_encoded = df.copy()

    # Create a LabelEncoder object to encode non-categorical string data
    label_encoder = LabelEncoder()

    # Iterate over each column in the DataFrame
    for column in df_encoded.columns:
        # Check if the column contains non-numeric data
        if df_encoded[column].dtype == 'object':
            # Check if the column contains categorical data
            if df_encoded[column].nunique() < len(df_encoded) / 2:
                # Use one-hot encoding to convert the column into a numerical format
                df_encoded = pd.get_dummies(df_encoded, columns=[column])
            else:
                # Use label encoding to convert the column into a numerical format
                df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

    return df_encoded


def encode_series(series):
    # Create a copy of the input Series to avoid modifying the original data
    series_encoded = series.copy()

    # Create a LabelEncoder object to encode non-categorical string data
    label_encoder = LabelEncoder()

    # Check if the Series contains non-numeric data
    if series_encoded.dtype == 'object':
        # Check if the Series contains categorical data
        if series_encoded.nunique() < len(series_encoded) / 2:
            # Use one-hot encoding to convert the Series into a numerical format
            series_encoded = pd.get_dummies(series_encoded)
        else:
            # Use label encoding to convert the Series into a numerical format
            series_encoded = pd.Series(label_encoder.fit_transform(series_encoded))

    return series_encoded



class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert data to numpy array and then to tensor
        x = torch.tensor(self.X.iloc[idx].to_numpy())
        y = torch.tensor(self.y.iloc[idx])

        # Reshape target data into a 0D tensor
        y = y.argmax()

        return x, y
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        return out