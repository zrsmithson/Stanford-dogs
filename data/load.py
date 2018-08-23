import os

from torchvision import transforms, datasets
from data.stanford_dogs_data import dogs
from data.oxford_flowers import flowers
import configs

def load_datasets(set_name, input_size=224):
    if set_name == 'mnist':
        train_dataset = datasets.MNIST(root=os.path.join(configs.imagesets, 'MNIST'),
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.MNIST(root=os.path.join(configs.imagesets, 'MNIST'),
                                                  train=False,
                                                  transform=transforms.ToTensor())

    elif set_name == 'stanford_dogs':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = dogs(root=configs.imagesets,
                                 train=True,
                                 cropped=False,
                                 transform=input_transforms,
                                 download=True)
        test_dataset = dogs(root=configs.imagesets,
                                train=False,
                                cropped=False,
                                transform=input_transforms,
                                download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    elif set_name == 'oxford_flowers':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = flowers(root=configs.imagesets,
                                  train=True,
                                  val=False,
                                  transform=input_transforms,
                                  download=True)
        test_dataset = flowers(root=configs.imagesets,
                                train=False,
                                val=True,
                                transform=input_transforms,
                                download=True)

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()
    else:
        return None, None

    return train_dataset, test_dataset, classes
