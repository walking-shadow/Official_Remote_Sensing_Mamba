import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BasicDataset(Dataset):
    """ Basic dataset for train, evaluation and test.
    
    Attributes:
        images_dir(str): path of images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        ids(list): name list of images.
        train_transforms_all(class): data augmentation applied to image and label.

    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        """ Init of basic dataset.
        
        Parameter:
            images_dir(str): path of images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.

        """

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.train_transforms_all = A.Compose([
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # 使用最简单的数据增强方法
            # A.Rotate(45, p=0.3),
            # A.ShiftScaleRotate(p=0.3),
        ], additional_targets={'image1': 'image'})

        self.normalize = A.Compose([
            A.Normalize()
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ])

    def __len__(self):
        """ Return length of dataset."""
        return len(self.ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label[label != 0] = 1
        return label
    
    @classmethod
    def grayscale_label_preprocess(cls, label):
        "To convert grayscale values to class indices"
        label[label == 85] = 1
        label[label == 170] = 2
        return label
    

    @classmethod
    def load(cls, filename):
        """Open image and convert image to array."""

        img = Image.open(filename)
        img = np.array(img).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        """ Index dataset.

        Index image name list to get image name, search image in image path with its name,
        open image and convert it to array.

        Preprocess array, apply data augmentation and noise addition(optional) on it, and convert array to tensor.

        Parameter:
            idx(int): index of dataset.

        Return:
            tensor(tensor): tensor of image.
            label_tensor(tensor): tensor of label.
            name(str): the same name of image and label.
        """


        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        label_file = list(self.labels_dir.glob(name + '.*'))

        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {name}: {label_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])

        label = self.load(label_file[0])
        label = self.grayscale_label_preprocess(label)

        if self.train:
            sample = self.train_transforms_all(image=img, mask=label)
            img, label = sample['image'], sample['mask']


        img = self.normalize(image=img)['image']
        sample = self.to_tensor(image=img, mask=label)
        # ipdb.set_trace()
        tensor, label_tensor = sample['image'].contiguous(), sample['mask'].contiguous()

        return tensor, label_tensor, name
