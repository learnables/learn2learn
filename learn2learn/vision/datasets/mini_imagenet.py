from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import requests
import zipfile
import os
import numpy as np
import shutil
import os


def mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                


class MiniImageNet(Dataset):
    def __init__(self, subset, transform=None, target_transform=None, data_path ='./data/'):
        """Dataset class representing miniImageNet dataset
        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
                 
        file_id = '0B3Irx3uQNoBMQ1FlNXJsZUdYWEE'
        self.data_path = data_path
        destination_for_zip = self.data_path + '/miniImageNet.zip'
        destination_to_extract = self.data_path + '/miniImageNet/images'
        mkdir(self.data_path + '/miniImageNet/images_background')
        mkdir(self.data_path + '/miniImageNet/images_evaluation')


        self.transform = transform
        self.target_transform = target_transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Resize(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        if not os.path.exists(destination_to_extract):
            os.makedirs(destination_to_extract)
            download_file_from_google_drive(file_id, destination_for_zip)
            with zipfile.ZipFile(destination_for_zip, 'r') as zip_ref:
                zip_ref.extractall(destination_to_extract)
            
            # Clean up folders
            mkdir(self.data_path + '/miniImageNet/images_background')
            mkdir(self.data_path + '/miniImageNet/images_evaluation')

            # Find class identities
            classes = []
            for root, _, files in os.walk(self.data_path + '/miniImageNet/images/'):
                for f in files:
                    if f.endswith('.jpg'):
                        classes.append(f[:-12])

            classes = list(set(classes))

            # Train/test split
            np.random.seed(0)
            np.random.shuffle(classes)
            background_classes, evaluation_classes = classes[:80], classes[80:]

            # Create class folders
            for c in background_classes:
                mkdir(self.data_path+ f'/miniImageNet/images_background/{c}/')

            for c in evaluation_classes:
                mkdir(self.data_path + f'/miniImageNet/images_evaluation/{c}/')

            # Move images to correct location
            for root, _, files in os.walk(self.data_path + '/miniImageNet/images'):
                for f in files:
                    if f.endswith('.jpg'):
                        class_name = f[:-12]
                        image_name = f[-12:]
                        # Send to correct folder
                        subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
                        src = f'{root}/{f}'
                        dst = self.data_path + f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
                        shutil.copy(src, dst)
        else:
            print("Folders already exist")
        
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']


    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        if self.target_transform == None:
            label = self.datasetid_to_class_id[item]
        else:
            label = self.datasetid_to_class_id[item]
            label = self.target_transform(label)
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def index_subset(self, sbs):
        """Index a subset by looping through all of its files and recording relevant information.
        # Arguments
            subset: Name of the subset
        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(sbs))
        subset_len = 0
        for root, folders, files in os.walk(self.data_path + '/miniImageNet/images_{}/'.format(sbs)):
            subset_len += len([f for f in files if f.endswith('.png')])

        for root, folders, files in os.walk(self.data_path + '/miniImageNet/images_{}/'.format(sbs)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                images.append({
                    'subset': sbs,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        return images

# Instructions:

# import learn2learn as l2l
# from torch.utils.data import Dataset, ConcatDataset


# data_path = '/datadrive/test/'
# minimagenet_background = l2l.vision.datasets.MiniImageNet(subset ='background', data_path=data_path)
# minimagenet_evaluation = l2l.vision.datasets.MiniImageNet(subset ='evaluation', data_path=data_path)
# minimagenet = ConcatDataset((minimagenet_background, minimagenet_evaluation))
# minimagenet = l2l.data.MetaDataset(minimagenet)


# eval_generator = l2l.data.TaskGenerator(dataset=minimagenet, ways=5)
# support_t = eval_generator.sample(shots=8)
# query_t = eval_generator.sample(shots=8)