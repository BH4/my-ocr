"""
Starting with the NIST data sorted by class, separate images into train, test,
and validation sets. Each set will be in its own folder and images within those
folders will be sorted by class.

After choosing the proportions for each set, the images will be split within
classes instead of as a whole. This prevents accidental over or under
representation of any class.

The data as downloaded repeats images from the hsf folders in the train
folders, but in some classes the hsf folders to not contain all of the images
in the train folders. Here I only use images in the train folders to avoid
duplicates and for simplicity.

Also scale images down to desired size before saving.

Second step in the pre-processing phase is to read images into numpy format
and save the resulting files as hdf5 objects.
"""
import os
import random
from shutil import copyfile, rmtree
from PIL import Image

import numpy as np
import h5py
import tensorflow as tf


def num_training_images(class_folder_path):
    """
    Output the number of images in each train_?? folder.
    """
    for root, dirs, files in os.walk(class_folder_path):
        folder_name = root.split('\\')[-1]
        if folder_name[:-2] == 'train_':
            print('Class {}: {} images'.format(folder_name[-2:], len(files)))


def make_copy(src, dst, size=None):
    if size is None:
        copyfile(src, dst)
    else:
        img = Image.open(src)
        img = img.resize((size, size), Image.ANTIALIAS)
        img.save(dst)


def train_test_split(class_folder_path, data_path, train, val, new_size=32, max_train=None, seed=None):
    """
    Delete all files in the folders
    data_path+'\\train'
    data_path+'\\validation'
    data_path+'\\test'

    And then randomly allocate images from the train_?? folders to these in
    proportion with the values given for train and val

    Test files will be remaining files after allocating train and validation
    sets. For example, if train=.8 and val=.1 then there 10% of the files will
    be test files.
    """
    if seed is not None:
        random.seed(seed)

    # Delete old files
    folders = ['\\train', '\\validation', '\\test']
    for folder in folders:
        if os.path.exists(data_path+folder):
            # os.rmdir(data_path+folder)
            print('Removing', data_path+folder)
            rmtree(data_path+folder)
        os.mkdir(data_path+folder)

    # Copy in new distribution
    for root, dirs, files in os.walk(class_folder_path):
        folder_name = root.split('\\')[-1]
        if folder_name[:-2] == 'train_':
            class_name = folder_name[-2:]

            print('Writing data for class {}'.write(class_name))

            # Randomize order
            random.shuffle(files)

            num_train = int(len(files)*train)
            num_val = int(len(files)*val)
            num_test = len(files)-num_train-num_val

            if max_train is not None and num_train > max_train:
                # keep_frac = max_train/float(num_train)
                num_train = max_train
                num_val = val*(max_train/train)  # int(keep_frac*num_val)
                num_test = (1-num_val-num_test)*(max_train/train)  # int(keep_frac*num_test)

            splits = [(0, num_train), (num_train, num_train+num_val),
                      (num_train+num_val, num_train+num_val+num_test)]

            for i in range(3):
                folder = folders[i]
                split = splits[i]

                # Create destination folder
                dst_folder = data_path+folder+'\\'+class_name
                os.mkdir(dst_folder)

                for f in files[split[0]:split[1]]:
                    src = root+'\\'+f
                    dst = dst_folder+'\\'+f
                    # copyfile(src, dst)
                    make_copy(src, dst, size=new_size)


def process_images(path, filename):
    """
    Process images into a numpy array to be saved as an hdf5 file.

    Assumes all classes are present in each of the train, validation, and test
    set. If not the categorical labels will not match between them.
    """
    X = []
    Y = []
    class_map = None
    prev_class = ''
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            # In data folder with class folders.
            # Get all classes, sort them, map to nums 0 to class_num
            class_list = sorted(dirs)
            class_map = {c: i for i, c in enumerate(class_list)}
        elif len(files) > 0:
            class_name = root.split('\\')[-1]
            if class_name != prev_class:
                prev_class = class_name
                print('Loading class {}. Found {} files.'.format(class_name, len(files)))
            for f in files:
                img_path = root+'\\'+f
                image = Image.open(img_path)
                x = np.array(image)[:, :, 0]  # r, g, and b channels are identical
                X.append(x)
                Y.append(class_map[class_name])

    X = np.array(X)/255.0
    X = X[:, :, :, np.newaxis]
    Y = tf.keras.utils.to_categorical(Y, len(class_map))

    hf = h5py.File(filename, 'w')
    hf.create_dataset('X', data=X)
    hf.create_dataset('Y', data=Y)
    hf.close()


if __name__ == '__main__':
    class_folder_path = '..\\data\\default_form\\by_class'
    # num_training_images(class_folder_path)

    """
    train = 0.8
    val = 0.1
    data_path = '..\\data'
    train_test_split(class_folder_path, data_path, train, val, new_size=32,  max_train=6000, seed=4)
    """

    print('Starting final train processing.')
    process_images('..\\data\\train', '..\\data\\processed\\train.h5')
    print('Starting final validation processing.')
    process_images('..\\data\\validation', '..\\data\\processed\\validation.h5')
    print('Starting final test processing.')
    process_images('..\\data\\test', '..\\data\\processed\\test.h5')
    print('Finished')
