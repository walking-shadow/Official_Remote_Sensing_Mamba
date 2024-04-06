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
        t1_images_dir(str): path of t1 images.
        t2_images_dir(str): path of t2 images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        t1_ids(list): name list of t1 images.
        t2_ids(list): name list of t2 images.
        train_transforms_all(class): data augmentation applied to t1 image, t2 image and label.
        train_transforms_image(class): noise addition only applied to t1 image and t2 image.


    """

    def __init__(self, t1_images_dir: str, t2_images_dir: str, labels_dir: str, train: bool,
                 ):
        """ Init of basic dataset.
        
        Parameter:
            t1_images_dir(str): path file of t1 images.
            t2_images_dir(str): path file of t2 images.
            labels_dir(str): path file of labels.
            train(bool): ensure creating a train dataset or other dataset.

        """

        self.t1_images_dir = Path(t1_images_dir)
        self.t2_images_dir = Path(t2_images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix
        self.t1_ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith('.')]
        self.t2_ids = [splitext(file)[0] for file in listdir(t2_images_dir) if not file.startswith('.')]
        self.t1_ids.sort()
        self.t2_ids.sort()

        if not self.t1_ids:
            raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
        if not self.t2_ids:
            raise RuntimeError(f'No input file found in {t2_images_dir}, make sure you put your images there')
        assert len(self.t1_ids) == len(self.t2_ids), 'number of t1 images is not equivalent to number of t2 images'
        logging.info(f'Creating dataset with {len(self.t1_ids)} examples')

        self.train_transforms_all = A.Compose([
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # 使用最简单的数据增强方法
            # A.Rotate(45, p=0.3),
            # A.ShiftScaleRotate(p=0.3),
        ], additional_targets={'image1': 'image'})

        self.t1_normalize = A.Compose([
            A.Normalize()
        ])

        self.t2_normalize = A.Compose([
            A.Normalize()
        ])

        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image'})

    def __len__(self):
        """ Return length of dataset."""
        return len(self.t1_ids)

    @classmethod
    def label_preprocess(cls, label):
        """ Binaryzation label."""

        label[label != 0] = 1
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

        Preprocess array, apply data augmentation and noise addition(optional) on it,
        random exchange t1 and t2 array, and convert array to tensor.

        Parameter:
            idx(int): index of dataset.

        Return:
            t1_tensor(tensor): tensor of t1 image.
            t2_tensor(tensor): tensor of t2 image.
            label_tensor(tensor): tensor of label.
            name(str): the same name of t1 image, t2 image and label.
        """


        t1_name = self.t1_ids[idx]
        t2_name = self.t2_ids[idx]
        assert t1_name == t2_name, f't1 name{t1_name} not equal to t2 name{t2_name}'
        t1_img_file = list(self.t1_images_dir.glob(t1_name + '.*'))
        t2_img_file = list(self.t2_images_dir.glob(t2_name + '.*'))
        label_file = list(self.labels_dir.glob(t1_name + '.*'))

        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {t1_name}: {label_file}'
        assert len(t1_img_file) == 1, f'Either no image or multiple images found for the ID {t1_name}: {t1_img_file}'
        t1_img = self.load(t1_img_file[0])
        t2_img = self.load(t2_img_file[0])

        label = self.load(label_file[0])
        label = self.label_preprocess(label)

        if self.train:
            sample = self.train_transforms_all(image=t1_img, image1=t2_img, mask=label)
            t1_img, t2_img, label = sample['image'], sample['image1'], sample['mask']
            # sample = self.train_transforms_image(image=t1_img, image1=t2_img)
            # t1_img, t2_img = sample['image'], sample['image1']

        t1_img = self.t1_normalize(image=t1_img)['image']
        t2_img = self.t2_normalize(image=t2_img)['image']
        if self.train:
            # random exchange t1_img and t2_img
            if random.choice([0, 1]):
                t1_img, t2_img = t2_img, t1_img
        sample = self.to_tensor(image=t1_img, image1=t2_img, mask=label)
        # ipdb.set_trace()
        t1_tensor, t2_tensor, label_tensor = sample['image'].contiguous(),\
                                             sample['image1'].contiguous(), sample['mask'].contiguous()
        name = t1_name

        return t1_tensor, t2_tensor, label_tensor, name
