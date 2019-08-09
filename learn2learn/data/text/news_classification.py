from torch.utils.data import Dataset
import pandas as pd
import torch
import requests
import zipfile
import io
import os


class NewsClassification(Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        self.labels_list = {'QUEER VOICES': 0, 'GREEN': 1, 'STYLE': 2, 'BUSINESS': 3, 'CULTURE & ARTS': 4,
                            'WEDDINGS': 5, 'ARTS': 6, 'HEALTHY LIVING': 7,
                            'LATINO VOICES': 8, 'ENVIRONMENT': 9, 'FIFTY': 10, 'COMEDY': 11, 'BLACK VOICES': 12,
                            'TRAVEL': 13, 'ENTERTAINMENT': 14, 'TASTE': 15,
                            'CRIME': 16, 'WOMEN': 17, 'TECH': 18, 'PARENTING': 19, 'SCIENCE': 20, 'WORLD NEWS': 21,
                            'WORLDPOST': 22, 'POLITICS': 23,
                            'ARTS & CULTURE': 24, 'RELIGION': 25, 'IMPACT': 26, 'MEDIA': 27, 'STYLE & BEAUTY': 28,
                            'SPORTS': 29, 'WEIRD NEWS': 30,
                            'HOME & LIVING': 31, 'THE WORLDPOST': 32, 'MONEY': 33, 'EDUCATION': 34, 'DIVORCE': 35,
                            'PARENTS': 36, 'GOOD NEWS': 37,
                            'FOOD & DRINK': 38, 'WELLNESS': 39, 'COLLEGE': 40}
        if train:
            self.path = os.path.join(root, 'train_sample.csv')
        else:
            self.path = os.path.join(root, 'test_sample.csv')
        self.transform = transform
        if transform == 'roberta':
            self.transform_func = torch.hub.load('pytorch/fairseq', 'roberta.large')

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
        if self.transform:
            return self.transform_func.extract_features(self.transform_func.encode(self.df_data['headline'][idx])), \
                   self.labels_list[self.df_data['category'][idx]]

        return self.df_data['headline'][idx], self.labels_list[self.df_data['category'][idx]]


train_dataset = NewsClassification(root='/tmp/dst/', transform='roberta', train=False, download=True)
