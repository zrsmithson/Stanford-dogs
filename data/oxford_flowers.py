from __future__ import print_function

from PIL import Image
from os.path import join
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files


class flowers(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'OxfordFlowers'
    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'

    def __init__(self,
                 root,
                 train=True,
                 val=False,
                 transform=None,
                 target_transform=None,
                 download=False,
                 classes=None):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.val = val
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.split = self.load_split()
        # self.split = self.split[:100]  # TODO: debug only get first ten classes

        self.images_folder = join(self.root, 'jpg')

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name, target_class = self.split[index]
        image_path = join(self.images_folder, "image_%05d.jpg" % (image_name+1))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'jpg')) and os.path.exists(join(self.root, 'imagelabels.mat')) and os.path.exists(join(self.root, 'setid.mat')):
            if len(os.listdir(join(self.root, 'jpg'))) == 8189:
                print('Files already downloaded and verified')
                return

        filename = '102flowers'
        tar_filename = filename + '.tgz'
        url = self.download_url_prefix + '/' + tar_filename
        download_url(url, self.root, tar_filename, None)
        with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, self.root)
        os.remove(join(self.root, tar_filename))

        filename = 'imagelabels.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

        filename = 'setid.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

    def load_split(self):
        split = scipy.io.loadmat(join(self.root, 'setid.mat'))
        labels = scipy.io.loadmat(join(self.root, 'imagelabels.mat'))['labels']
        if self.train:
            split = split['trnid']
        elif self.val:
            split = split['valid']
        else:
            split = split['tstid']

        split = list(split[0] - 1) # set it all back 1 as img indexs start at 1
        labels = list(labels[0][split]-1)
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.split)):
            image_name, target_class = self.split[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self.split), len(counts.keys()), float(len(self.split))/float(len(counts.keys()))))

        return counts
