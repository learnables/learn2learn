## Code for separation into background and evaluation adapted from 

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
                

if __name__ == "__main__":
    
    DATA_PATH = '' ## Set this to the folder you want data to be downloaded to
    
    if not DATA_PATH:
        raise ValueError("DATA PATH must be set")
    
    
    file_id = '0B3Irx3uQNoBMQ1FlNXJsZUdYWEE'
    
    destination_for_zip = '/datadrive/test/miniImageNet.zip'
    destination_to_extract = '/datadrive/test/miniImageNet/images'

    if not os.path.exists(destination_to_extract):
        os.makedirs(destination_to_extract)
        download_file_from_google_drive(file_id, destination_for_zip)
        with zipfile.ZipFile(destination_for_zip, 'r') as zip_ref:
            zip_ref.extractall(destination_to_extract)
        # Clean up folders
        mkdir(DATA_PATH + '/miniImageNet/images_background')
        mkdir(DATA_PATH + '/miniImageNet/images_evaluation')

        # Find class identities
        classes = []
        for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
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
            mkdir(DATA_PATH + f'/miniImageNet/images_background/{c}/')

        for c in evaluation_classes:
            mkdir(DATA_PATH + f'/miniImageNet/images_evaluation/{c}/')

        # Move images to correct location
        for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images'):
            for f in files:
                if f.endswith('.jpg'):
                    class_name = f[:-12]
                    image_name = f[-12:]
                    # Send to correct folder
                    subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
                    src = f'{root}/{f}'
                    dst = DATA_PATH + f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
                    shutil.copy(src, dst)
    else:
        print("Folders already exist")
