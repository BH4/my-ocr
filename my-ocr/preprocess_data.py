"""
Starting with the NIST data (zip file or folder containing the unzipped
directory), process images and create two hdf5 files. A training file and
a testing file.

Specify:
- NIST data location (zip file or folder containing the unzipped directory)
- The maximum number of samples from a single class
- The proportion of the maximum number which should be classified as training
- The proportion of the maximum number which should be classified as validation
- Seed for random selections.
- Path and base name for saved data.
- Optional: list of classes which should be merged and treated as the same class

This program will:
- Read all images from NIST 'train' folders
- Process each image
- Randomly select a number of images corresponding to the maximum number of
  samples from each of the (merged) classes
- Randomly divide images between training, validation, and testing
- Create two hdf5 files. One containing training and validation images and the
  other containing testing images.

Processing includes:
- Remove color axis since images are black and white.
- Normalize pixel values to either 0 or 1

Note smallest class is 0x6a (which is lowercase j) with 1920 samples.
"""

import os
import numpy as np
import h5py
import tensorflow as tf

from image_reader import data_reader


def process_images(data_path, max_samples, train, val, seed, save_path, save_name, combined_classes_hex=[]):
    merge_map = {}
    for merge in combined_classes:
        for c in merge:
            merge_map[c] = merge

    # List of all the hex values corresponding to all 62 characters in the set
    full_class_hex_list = []
    for i in range(10):
        full_class_hex_list.append(hex(ord('0')+i)[2:])
    for i in range(26):
        full_class_hex_list.append(hex(ord('A')+i)[2:])
    for i in range(26):
        full_class_hex_list.append(hex(ord('a')+i)[2:])

    # Read all files and organize into dictionary labeled by the classes hex descriptor
    all_images = {}
    with data_reader(data_path) as z:
        for class_hex, img in z.relevant_file_walk():
            # img is black and white (128, 128, 3)
            # Removing color channel and normalizing pixel value to 0 or 1
            img = img[:, :, 0]//255

            if class_hex not in all_images:
                all_images[class_hex] = [img]
            else:
                all_images[class_hex].append(img)

    # train_count = int(train*max_samples)
    # val_count = int(val*max_samples)
    # test_count = max_samples-train_count-val_count

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    # Randomize image orders
    rng = np.random.default_rng(seed)
    for c_hex in full_class_hex_list:
        rng.shuffle(all_images[c_hex])

    # Save max_sample number of images from each class
    class_map = {}
    curr_label = 0
    for c_hex in full_class_hex_list:
        if c_hex not in class_map:
            if c_hex not in merge_map:
                # Class is not merged with any others
                class_map[c_hex] = curr_label

                # Note, number of samples may be less than max_samples if there
                # are fewer total images
                randomized_samples = all_images[c_hex][:max_samples]
            else:
                randomized_samples = []

                merge = merge_map[c_hex]
                # Limit merged classes to have the same number of samples from each
                min_class_size = min([len(all_images[c_hex_i]) for c_hex_i in merge])
                merge_samples = min(max_samples, min_class_size*len(merge))
                if max_samples < min_class_size*len(merge):
                    subclass_samples = max_samples//len(merge)
                else:
                    subclass_samples = min_class_size

                for ind, c_hex_i in enumerate(merge):
                    class_map[c_hex_i] = curr_label

                    randomized_samples.extend(all_images[c_hex_i][:subclass_samples])

                # Shuffle newly merged samples
                rng.shuffle(randomized_samples)

            # Separate samples into training, validation, and test
            num_samples = len(randomized_samples)
            train_count = int(train*num_samples)
            val_count = int(val*num_samples)
            test_count = num_samples-train_count-val_count

            train_images.extend(randomized_samples[:train_count])
            val_images.extend(randomized_samples[train_count:train_count+val_count])
            test_images.extend(randomized_samples[train_count+val_count:])

            train_labels.extend([curr_label]*train_count)
            val_labels.extend([curr_label]*val_count)
            test_labels.extend([curr_label]*test_count)

            curr_label += 1

    number_of_labels = curr_label

    # Convert data and labels before saving
    train_images = np.array(train_images)
    val_images = np.array(val_images)
    test_images = np.array(test_images)
    train_labels = tf.keras.utils.to_categorical(train_labels, number_of_labels)
    val_labels = tf.keras.utils.to_categorical(val_labels, number_of_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels, number_of_labels)

    # Save data to separate training and testing h5 files
    train_name = os.path.join(save_path, save_name+'_train.h5')
    with h5py.File(train_name, 'w') as hf:
        hf.create_dataset('train_data', data=train_images)
        hf.create_dataset('train_label', data=train_labels)
        hf.create_dataset('val_data', data=val_images)
        hf.create_dataset('val_label', data=val_labels)

    test_name = os.path.join(save_path, save_name+'_test.h5')
    with h5py.File(test_name, 'w') as hf:
        hf.create_dataset('test_data', data=test_images)
        hf.create_dataset('test_label', data=test_labels)

    # Save map describing class labeling with the data.
    class_map_name = os.path.join(save_path, save_name+'_class_map.txt')
    with open(class_map_name, 'w') as f:
        f.write(str(class_map))


if __name__ == '__main__':
    """
    - NIST data location (zip file or folder containing the unzipped directory)
    - The maximum number of samples from a single class
    - The proportion of the maximum number which should be classified as training
    - The proportion of the maximum number which should be classified as validation
    - Seed for random selections.
    - Path for saved data
    - Base name for saved data
    - Optional: list of classes which should be merged and treated as the same class
    """
    ###########################################################################
    # Input
    ###########################################################################
    data_path = '../data/by_class.zip'
    max_samples = 2000
    train = 0.8
    val = 0.1
    # test fraction = 1-train-val
    seed = 4444
    save_path = '../data/processed'

    save_name = f'full_62_classes_seed{seed}'
    combined_classes = []
    # save_name = 'capital_merge_seed{seed}'
    # combined_classes = [('c', 'C'), ('k', 'K'), ('m', 'M'), ('p', 'P'),
    #                     ('s', 'S'), ('u', 'U'), ('v', 'V'), ('w', 'W'),
    #                     ('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('o', 'O')]
    # save_name = 'capital_and_zero_merge_seed{seed}'
    # combined_classes = [('c', 'C'), ('k', 'K'), ('m', 'M'), ('p', 'P'),
    #                     ('s', 'S'), ('u', 'U'), ('v', 'V'), ('w', 'W'),
    #                     ('x', 'X'), ('y', 'Y'), ('z', 'Z'), ('o', 'O', '0')]
    ###########################################################################

    # Convert class names to hex values
    combined_classes_hex = []
    for combo in combined_classes:
        combo_hex = tuple([hex(ord(x))[2:] for x in combo])
        combined_classes_hex.append(combo_hex)

    print('Starting processing')
    process_images(data_path, max_samples, train, val, seed, save_path, save_name, combined_classes_hex)
    print('Finished')
