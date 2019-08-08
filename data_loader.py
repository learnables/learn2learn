from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import requests
import zipfile
import io
import os

class TextClassification(Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        if train:
            self.path = os.path.join(root, 'train_sample.csv')
        else:
            self.path = os.path.join(root, 'test_sample.csv')
        self.transform = transform
        
        if download:
            download_file_url = 'https://www.dropbox.com/s/g8hwl9pxftl36ww/test_sample.csv.zip?dl=1'
            if train:
                download_file_url = 'https://www.dropbox.com/s/o71z7fq7mydbznc/train_sample.csv.zip?dl=1'
            
            r = requests.get(download_file_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=root)
        
        if root:
            
            if os.path.exists(self.path):
                self.df_data = pd.read_csv(self.path)
            else:
                raise ValueError("Please download the file first.")
                

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        return (self.df_data['headline'][idx], self.df_data['category'][idx])
    
train_dataset = TextClassification(root='/tmp/dst/',transform=None, train=False, download=True)