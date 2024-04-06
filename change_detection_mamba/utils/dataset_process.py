import sys
import numpy as np
from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import time
import wandb
import shutil


def verify_correspondence(dataset_name, mode=None):
    """ Verify correspondence between train/val/test dataset.

    Make sure there are corresponding images with the same name in :obj:`t1_images_dir`,
    :obj:`t2_images_dir` and :obj:`label_images_dir`.

    Notice that if mode is None,
    image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/ , else
    be organized as :obj:`dataset_name`/:obj:`mode`/`t1` or `t2` or `label`/ .

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether verifying train, val or test dataset.

    Return:
        return correspondence verified result.
    """

    if mode is None:
        images_dir = {'t1_images_dir': Path(f'./{dataset_name}/t1/'),
                      't2_images_dir': Path(f'./{dataset_name}/t2/'),
                      'label_images_dir': Path(f'./{dataset_name}/label/')
                      }
    else:
        images_dir = {'t1_images_dir': Path(f'./{dataset_name}/{mode}/t1/'),
                      't2_images_dir': Path(f'./{dataset_name}/{mode}/t2/'),
                      'label_images_dir': Path(f'./{dataset_name}/{mode}/label/')
                      }
    image_names = []
    for dir_path in images_dir.values():
        image_name = [splitext(file)[0] for file in listdir(dir_path) if not file.startswith('.')]
        image_names.append(image_name)
    image_names = np.unique(np.array(image_names))
    if len(image_names) != 1:
        print('Correspondence False')
        return False
    else:
        print('Correspondence Verified')
        return True


def delete_monochrome_image(dataset_name, mode=None):
    """ Delete monochrome images in dataset.

    Delete whole black and whole white image in label directory
    and corresponding image in t1 and t2 directory.

    Notice that if mode is None,
    image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/ , else
    be organized as :obj:`dataset_name`/:obj:`mode`/`t1` or `t2` or `label`/ .

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether verifying train, val or test dataset.

    Return:
        return nothing
    """

    if mode is None:
        t1_images_dir = Path(f'./{dataset_name}/t1/')
        t2_images_dir = Path(f'./{dataset_name}/t2/')
        label_images_dir = Path(f'./{dataset_name}/label/')
    else:
        t1_images_dir = Path(f'./{dataset_name}/{mode}/t1/'),
        t2_images_dir = Path(f'./{dataset_name}/{mode}/t2/'),
        label_images_dir = Path(f'./{dataset_name}/{mode}/label/')

    ids = [splitext(file)[0] for file in listdir(t1_images_dir) if not file.startswith('.')]
    img_name_sample = listdir(t1_images_dir)[0]
    img_sample = Image.open(str(t1_images_dir) + str(img_name_sample))
    img_size = img_sample.size[0]  # the image's height and width should be same

    if not ids:
        raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
    for name in tqdm(ids):
        label_img_dir = list(label_images_dir.glob(str(name) + '.*'))
        assert len(label_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {label_img_dir}'
        img = Image.open(label_img_dir[0])
        img_array = np.array(img)
        array_sum = np.sum(img_array)
        if array_sum == 0 or array_sum == (255 * img_size * img_size):
            path = label_img_dir[0]
            path.unlink()

            t1_img_dir = list(t1_images_dir.glob(str(name) + '.*'))
            assert len(t1_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {t1_img_dir}'
            path = t1_img_dir[0]
            path.unlink()

            t2_img_dir = list(t2_images_dir.glob(str(name) + '.*'))
            path = t2_img_dir[0]
            path.unlink()
    print('Over')


def compute_mean_std(images_dir):
    """Compute the mean and std of dataset images.

    Parameter:
        dataset_name(str): name of the specified dataset.

    Return:
        means(list): means in three channel(RGB) of images in :obj:`images_dir`
        stds(list): stds in three channel(RGB) of images in :obj:`images_dir`
    """

    images_dir = Path(images_dir)
    # calculate means and std
    means = [0, 0, 0]
    stds = [0, 0, 0]

    ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
    num_imgs = len(ids)

    if not ids:
        raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
    for name in tqdm(ids):
        img_file = list(images_dir.glob(str(name) + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = Image.open(img_file[0])
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img_array[:, :, i].mean()
            stds[i] += img_array[:, :, i].std()

    means = np.asarray(means) / num_imgs
    stds = np.asarray(stds) / num_imgs

    print("normMean = {}".format(means))
    print("normStd = {}".format(stds))

    return means, stds


def crop_img(dataset_name, pre_size, after_size, overlap_size):
    """Crop dataset images.

    Crop image from :math:`pre_size` × :math:`pre_size` to
    :math:`after_size` × :math:`after_size` with :math:`overlap_size` overlap for train dataset,
    and without overlap for validation and test dataset.

    The :math:(:math:`pre_size` - :math:`after size`) should be multiple of
    :math:(:math:`after_size` - :math:`overlap_size`),
    and :math:`pre_size` should be multiple of :math:`after size`.

    Notice that image path should be organized as
    :obj:`dataset_name`/`train` or `val` or `test`/`t1` or `t2` or `label`/.

    Parameter:
        dataset_name(str): name of the specified dataset.
        pre_size(int): image size before crop.
        after_size(int): image size after crop.
        overlap_size(int): images overlap size while crop in train dataset.

    Return:
        return nothing.
    """

    if (pre_size - after_size % after_size - overlap_size != 0) or (pre_size % after_size != 0):
        print(f'ERROR: the pre_size - after size should be multiple of after_size - overlap_size, '
              f'and pre_size should be multiple of after size')
        sys.exit()

    # data path should be dataset_name/train or val or test/t1 or t2 or label
    train_image_dirs = [
        Path(f'./{dataset_name}/train/t1/'),
        Path(f'./{dataset_name}/train/t2/'),
        Path(f'./{dataset_name}/train/label/'),
    ]
    train_save_dirs = [
        f'./{dataset_name}_crop/train/t1/',
        f'./{dataset_name}_crop/train/t2/',
        f'./{dataset_name}_crop/train/label/',
    ]
    val_test_image_dirs = [
        Path(f'./{dataset_name}/val/t1/'),
        Path(f'./{dataset_name}/val/t2/'),
        Path(f'./{dataset_name}/val/label/'),
        Path(f'./{dataset_name}/test/t1/'),
        Path(f'./{dataset_name}/test/t2/'),
        Path(f'./{dataset_name}/test/label/'),
    ]
    val_test_save_dirs = [
        f'./{dataset_name}_crop/val/t1/',
        f'./{dataset_name}_crop/val/t2/',
        f'./{dataset_name}_crop/val/label/',
        f'./{dataset_name}_crop/test/t1/',
        f'./{dataset_name}_crop/test/t2/',
        f'./{dataset_name}_crop/test/label/',
    ]
    slide_size = after_size - overlap_size
    slide_times_with_overlap = (pre_size - after_size) // slide_size + 1
    slide_times_without_overlap = pre_size // after_size

    print('Start crop training images')
    # crop train images
    for images_dir, save_dir in zip(train_image_dirs, train_save_dirs):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ids = [splitext(file) for file in listdir(images_dir) if not file.startswith('.')]
        if not ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        for name, suffix in tqdm(ids):
            img_file = list(images_dir.glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            img = Image.open(img_file[0])
            for i in range(slide_times_with_overlap):
                for j in range(slide_times_with_overlap):
                    box = (i * slide_size, j * slide_size,
                           i * slide_size + after_size, j * slide_size + after_size)
                    region = img.crop(box)
                    region.save(save_dir + f'/{name}_{i}_{j}{suffix}')

    print('Start crop val and test images')
    # crop val and test images
    for images_dir, save_dir in zip(val_test_image_dirs, val_test_save_dirs):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ids = [splitext(file) for file in listdir(images_dir) if not file.startswith('.')]
        if not ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        for name, suffix in tqdm(ids):
            img_file = list(images_dir.glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            img = Image.open(img_file[0])
            for i in range(slide_times_without_overlap):
                for j in range(slide_times_without_overlap):
                    box = (i * after_size, j * after_size, (i + 1) * after_size, (j + 1) * after_size)
                    region = img.crop(box)
                    region.save(save_dir + f'/{name}_{i}_{j}{suffix}')
    print('Over')


def image_shuffle(dataset_name):
    """ Shuffle dataset images.

    Shuffle images in dataset to random split images to train, val and test later.

    Notice that image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/.

    Parameter:
        dataset_name(str): name of the specified dataset.

    Return:
        return nothing.
    """

    # data path should be dataset_name/t1 or t2 or label
    t1_images_dir = Path(f'./{dataset_name}/t1/')
    t2_images_dir = Path(f'./{dataset_name}/t2/')
    label_images_dir = Path(f'./{dataset_name}/label/')

    ids = [splitext(file) for file in listdir(t1_images_dir) if not file.startswith('.')]
    Imgnum = len(ids)
    L = random.sample(range(0, Imgnum), Imgnum)

    if not ids:
        raise RuntimeError(f'No input file found in {t1_images_dir}, make sure you put your images there')
    for i, (name, suffix) in tqdm(enumerate(ids)):
        t1_img_dir = list(t1_images_dir.glob(str(name) + '.*'))
        assert len(t1_img_dir) == 1, f'Either no mask or multiple masks found for the ID {name}: {t1_img_dir}'
        path = Path(t1_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)

        t2_img_dir = list(t2_images_dir.glob(str(name) + '.*'))
        path = Path(t2_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)

        label_img_dir = list(label_images_dir.glob(str(name) + '.*'))
        path = Path(label_img_dir[0])
        new_file = path.with_name('shuffle_' + str(L[i]) + str(suffix))
        path.replace(new_file)
    print('Over')


def split_image(dataset_name, fixed_ratio=True):
    """ Split dataset images.

    Split images to trian/val/test dataset with 7:2:1 ratio or corresponding specified number.

    Notice that image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/.

    Parameter:
        dataset_name(str): name of the specified dataset.
        fixed_ratio(bool): if True, split images with 7:2:1 ratio, else split with corresponding specified number,
            which should be set in this function.

    Return:
        return nothing.
    """
    source_image_dirs = [
        Path(f'./{dataset_name}/t1'),
        Path(f'./{dataset_name}/t2'),
        Path(f'./{dataset_name}/label'),
    ]
    train_save_dirs = [
        f'./{dataset_name}_split/train/t1/',
        f'./{dataset_name}_split/train/t2/',
        f'./{dataset_name}_split/train/label/',
    ]
    val_save_dirs = [
        f'./{dataset_name}_split/val/t1/',
        f'./{dataset_name}_split/val/t2/',
        f'./{dataset_name}_split/val/label/',
    ]
    test_save_dirs = [
        f'./{dataset_name}_split/test/t1/',
        f'./{dataset_name}_split/test/t2/',
        f'./{dataset_name}_split/test/label/',
    ]

    for i in range(3):
        Path(train_save_dirs[i]).mkdir(parents=True, exist_ok=True)
        Path(val_save_dirs[i]).mkdir(parents=True, exist_ok=True)
        Path(test_save_dirs[i]).mkdir(parents=True, exist_ok=True)

        ids = [splitext(file) for file in listdir(source_image_dirs[i]) if not file.startswith('.')]
        ids.sort()
        if not ids:
            raise RuntimeError(f'No input file found in {source_image_dirs[i]}, make sure you put your images there')

        if fixed_ratio:
            whole_num = len(ids)
            train_num = int(0.7 * whole_num)
            val_num = int(0.2 * whole_num)
            test_num = int(0.1 * whole_num)
        else:
            train_num = 540
            val_num = 152
            test_num = 1828

        for step, (name, suffix) in tqdm(enumerate(ids)):
            img_file = list(source_image_dirs[i].glob(str(name) + '.*'))
            assert len(img_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {img_file}'
            if step <= train_num:
                img_path = Path(img_file[0])
                new_path = Path(train_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
            elif step <= train_num + val_num:
                img_path = Path(img_file[0])
                new_path = Path(val_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
            else:
                img_path = Path(img_file[0])
                new_path = Path(test_save_dirs[i] + str(name) + str(suffix))
                img_path.replace(new_path)
    print('Over')


def crop_whole_image(dataset_name, crop_size):
    """ Crop whole large image.

    Crop the whole large image into :math:`crop_size`×:math:`crop_size` image without overlap.

    Notice source image path should be set in this function.

    Parameter:
        dataset_name(str): name of the specified dataset.
        crop_size(int): image size after crop.

    Return:
        return nothing.
    """

    Image.MAX_IMAGE_PIXELS = None
    # images_path and suffix should be set
    images_path = [Path('./njds/T1_img/2014.tif'),
                   Path('./njds/T2_img/2018.tif'),
                   Path('./njds/Change_Label/gt.tif')
                   ]
    suffix = '.tif'
    save_path = [f'./{dataset_name}/t1/',
                 f'./{dataset_name}/t2/',
                 f'./{dataset_name}/label/'
                 ]
    for path in save_path:
        Path(path).mkdir(parents=True, exist_ok=True)
    for n in tqdm(range(len(images_path))):
        image = Image.open(images_path[n])
        w, h = image.size
        print(f'image size: {image.size}')
        for j in range(w // crop_size + 1):
            for i in range(h // crop_size + 1):
                if i == h // crop_size:
                    y1 = h - crop_size
                    y2 = h
                else:
                    y1 = i * crop_size
                    y2 = (i + 1) * crop_size
                if j == w // crop_size:
                    x1 = w - crop_size
                    x2 = w
                else:
                    x1 = j * crop_size
                    x2 = (j + 1) * crop_size

                box = (x1, y1, x2, y2)
                region = image.crop(box)
                region.save(save_path[n] + f'/{j}_{i}{suffix}')


def compare_predset():
    """Compare two pred set and save their difference.

    Notice that path of two pred set should be set in this function.

    Parameter:
        nothing.

    Return:
        return nothing.
    """

    # two pred set should be set first
    pred_set_1 = Path('./njds_val_dedf_pred_dir')  # dedf path
    pred_set_2 = Path('./njds_val_ded_pred_dir')  # ded path

    step = 0
    difference_dict = {}

    ids = [splitext(file) for file in listdir(pred_set_1) if not file.startswith('.')]
    if not ids:
        raise RuntimeError(f'No input file found in {pred_set_1}, make sure you put your images there')
    for name, suffix in ids:
        step += 1
        print(f'step: {step}')
        pred_1_file = list(pred_set_1.glob(str(name) + '.*'))
        assert len(pred_1_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {pred_1_file}'
        pred_1_image = Image.open(pred_1_file[0])
        pred_1_array = np.array(pred_1_image)

        pred_2_file = list(pred_set_2.glob(str(name) + '.*'))
        assert len(pred_2_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {pred_2_file}'
        pred_2_image = Image.open(pred_2_file[0])
        pred_2_array = np.array(pred_2_image)

        difference = np.sum(np.abs(pred_2_array - pred_1_array))

        difference_dict[str(name)] = difference

    ordered_difference_list = sorted(difference_dict.items(), key=lambda x: x[1], reverse=True)
    np.save('njds_ordered_val_difference.npy', np.array(ordered_difference_list))
    print('Over')


def display_dataset_image(dataset_name, mode=None):
    """ Display dataset image in wandb to inspect images.

    Notice that if mode is None,
    image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/ , else
    be organized as :obj:`dataset_name`/:obj:`mode`/`t1` or `t2` or `label`/ .

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether sample train, val or test dataset.

    Return:
        return nothing.
    """

    if mode is not None:
        display_img_path = [f'./{dataset_name}/{mode}/t1/',
                            f'./{dataset_name}/{mode}/t2/',
                            f'./{dataset_name}/{mode}/label/'
                            ]
    else:
        display_img_path = [f'./{dataset_name}/t1/',
                            f'./{dataset_name}/t2/',
                            f'./{dataset_name}/label/'
                            ]
    localtime = time.asctime(time.localtime(time.time()))
    log_wandb = wandb.init(project='dpcd_last', resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=dict(time=localtime))
    ids = [splitext(file)[0] for file in listdir(display_img_path[0]) if not file.startswith('.')]
    if not ids:
        raise RuntimeError(f'No input file found in {display_img_path[0]}, make sure you put your images there')
    for name in tqdm(ids):
        display_img1 = list(Path(display_img_path[0]).glob(str(name) + '.*'))
        assert len(display_img1) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img1}'
        display_img1 = Image.open(display_img1[0])

        display_img2 = list(Path(display_img_path[1]).glob(str(name) + '.*'))
        assert len(display_img2) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img2}'
        display_img2 = Image.open(display_img2[0])

        display_img3 = list(Path(display_img_path[2]).glob(str(name) + '.*'))
        assert len(display_img3) == 1, f'Either no mask or multiple masks found for the ID {name}: {display_img3}'
        display_img3 = Image.open(display_img3[0])

        log_wandb.log({
            f'{display_img_path[0]}': wandb.Image(display_img1),
            f'{display_img_path[1]}': wandb.Image(display_img2),
            f'{display_img_path[2]}': wandb.Image(display_img3)
        })

    print('Over')


def sample_dataset(dataset_name, mode=None, ratio=None, num=None):
    """ Random sample specified ratio or number of dataset.

    Notice that if mode is None,
    image path should be organized as :obj:`dataset_name`/`t1` or `t2` or `label`/ , else
    be organized as :obj:`dataset_name`/:obj:`mode`/`t1` or `t2` or `label`/.

    Parameter:
        dataset_name(str): name of the specified dataset.
        mode(str): ensure whether sample train, val or test dataset.
        ratio(float): if not None, sample dataset with :math:`ratio` times :math:`dataset_size`.
        num(int): if not None, sample dataset with this num.
            if ratio and num are both not None, sample dataset with specified ratio.

    Return:
        return nothing.
    """

    if mode is not None:
        source_img_path = [
            f'./{dataset_name}/{mode}/t1/',
            f'./{dataset_name}/{mode}/t2/',
            f'./{dataset_name}/{mode}/label/'
        ]
        save_sample_img_path = [
            f'./{dataset_name}_sample/{mode}/t1/',
            f'./{dataset_name}_sample/{mode}/t2/',
            f'./{dataset_name}_sample/{mode}/label/'
        ]
    else:
        source_img_path = [
            f'./{dataset_name}/t1/',
            f'./{dataset_name}/t2/',
            f'./{dataset_name}/label/'
        ]
        save_sample_img_path = [
            f'./{dataset_name}_sample/t1/',
            f'./{dataset_name}_sample/t2/',
            f'./{dataset_name}_sample/label/'
        ]

    assert not (ratio is None and num is None), 'ratio and num are None at the same time'

    ids = [splitext(file) for file in listdir(source_img_path[0]) if not file.startswith('.')]
    Imgnum = len(ids)
    if ratio is not None:
        num = Imgnum * ratio
    img_index = random.sample(range(0, Imgnum), num)
    sample_imgs = [ids[i] for i in img_index]
    if not ids:
        raise RuntimeError(f'No input file found in {source_img_path[0]}, make sure you put your images there')
    for name, suffix in tqdm(sample_imgs):
        for i in range(len(source_img_path)):
            source_img = list(Path(source_img_path[i]).glob(str(name) + '.*'))
            assert len(source_img) == 1, f'Either no mask or multiple masks found for the ID {name}: {source_img}'
            source_file = Path(source_img[0])
            new_file = Path(save_sample_img_path[i] + str(name) + str(suffix))
            shutil.copyfile(source_file, new_file)
    print('Over')
