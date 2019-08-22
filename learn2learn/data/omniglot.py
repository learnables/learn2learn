import torch.utils.data as data
import os
import os.path
import errno

from PIL import Image

"""
Implementation based on https://github.com/dragen1860/MAML-Pytorch.
"""


class Omniglot(data.Dataset):

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip',
    ]
    zip_folder = 'zip'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be
    found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self,
                 root='./data/',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.'
                                   ' You can use download=True to download it')

        if not self._check_exists(folder=self.processed_folder):
            self.all_items = find_classes(os.path.join(self.root,
                                                       self.raw_folder))
            self.idx_classes = index_classes(self.all_items)
            self.process_raw()

        self.all_items = find_classes(os.path.join(self.root,
                                                   self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self, folder=None):
        if folder is None:
            folder = self.raw_folder
        has_evaluation = os.path.exists(os.path.join(self.root,
                                                     folder))
        has_background = os.path.exists(os.path.join(self.root,
                                                     folder,
                                                     ))
        return has_evaluation and has_background

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        self.mkdir(os.path.join(self.root, self.zip_folder))
        self.mkdir(os.path.join(self.root, self.raw_folder))

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.zip_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")

    def mkdir(self, path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def process_raw(self):
        """
        This function preprocesses the data in the same way
        C.B. Finn's code does.
        """
        self.mkdir(os.path.join(self.root, self.processed_folder))
        print('== Preprocessing Omniglot images, CBF-style.')
        dest_dir = os.path.join(self.root, self.processed_folder)
        for fname, class_name, fdir in self.all_items:
            path = os.path.join(fdir, fname)
            im = Image.open(path)
            im = im.resize((28, 28), resample=Image.LANCZOS)
            fdest = os.path.join(dest_dir, class_name)
            self.mkdir(fdest)
            im.save(os.path.join(fdest, fname))


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx
