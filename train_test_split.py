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
"""
import os
import random
from shutil import copyfile


def num_training_images(class_folder_path):
    """
    Output the number of images in each train_?? folder.
    """
    for root, dirs, files in os.walk(class_folder_path):
        folder_name = root.split('\\')[-1]
        if folder_name[:-2] == 'train_':
            print('Class {}: {} images'.format(folder_name[-2:], len(files)))


def train_test_split(class_folder_path, data_path, train, val, seed=None):
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
            os.rmdir(data_path+folder)
        os.mkdir(data_path+folder)

    # Copy in new distribution
    for root, dirs, files in os.walk(class_folder_path):
        folder_name = root.split('\\')[-1]
        if folder_name[:-2] == 'train_':
            class_name = folder_name[-2:]

            # Randomize order
            random.shuffle(files)

            num_train = int(len(files)*train)
            num_val = int(len(files)*val)
            # test is remaining proportion

            splits = [(0, num_train), (num_train, num_train+num_val),
                      (num_train+num_val, len(files))]

            for i in range(3):
                folder = folders[i]
                split = splits[i]

                # Create destination folder
                dst_folder = data_path+folder+'\\'+class_name
                os.mkdir(dst_folder)

                for f in files[split[0]:split[1]]:
                    src = root+'\\'+f
                    dst = dst_folder+'\\'+f
                    copyfile(src, dst)


if __name__ == '__main__':
    class_folder_path = '..\\data\\default_form\\by_class'
    # num_training_images(class_folder_path)

    train = 0.8
    val = 0.1
    data_path = '..\\data'

    train_test_split(class_folder_path, data_path, train, val, seed=4)
