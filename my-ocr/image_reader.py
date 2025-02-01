import os
import zipfile
import io
from PIL import Image
import numpy as np


class data_reader(object):
    """
    Given path to the zip file or a folder containing the images, acts as a
    context for reading them in sequence.
    """
    def __init__(self, path):
        self.path = os.path.normpath(path)

        if path[-4:] == '.zip':
            self.file = zipfile.ZipFile(self.path, mode='r')
            self.namelist_set = set(self.file.namelist())
        else:
            self.file = None
            self.namelist = None

    def __enter__(self):  # enter and exit for context management
        return self

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()

    def close(self):
        self.__exit__()

    def exists(self, path):
        if self.file is not None:
            return path in self.namelist_set
        else:
            full_path = os.path.join(self.path, path)
            return os.path.exists(full_path)

    def file_img_data(self, img_path):
        if self.file is not None:
            with self.file.open(img_path) as image_file:
                # Open the image using PIL from a BytesIO stream
                image = Image.open(io.BytesIO(image_file.read()))

                # Return the image as a numpy array
                return np.array(image)
        else:
            full_path = os.path.join(self.path, img_path)
            with Image.open(full_path) as image:
                # Return the image as a numpy array
                return np.array(image)

    def relevant_file_walk(self):
        """
        yield: (class_hex, numpy array representing the image)

        Files in the zip have paths
        by_class/hex/train_hex/train_hex_xxxxx.png

        Where "hex" is the hexidecimal value for the character represented by
        the class and xxxxx are numbers increasing from 00000.
        """
        # List hex values for each class (62 total)
        class_hex = []
        for i in range(10):
            class_hex.append(hex(ord('0')+i)[2:])
        for i in range(26):
            class_hex.append(hex(ord('A')+i)[2:])
        for i in range(26):
            class_hex.append(hex(ord('a')+i)[2:])

        count = 0
        for c in class_hex:
            count = 0
            count_str = str(count)
            count_str = '0'*(5-len(count_str))+count_str
            path = f'by_class/{c}/train_{c}/train_{c}_{count_str}.png'
            while self.exists(path):
                yield (c, self.file_img_data(path))

                count += 1
                count_str = str(count)
                count_str = '0'*(5-len(count_str))+count_str
                path = f'by_class/{c}/train_{c}/train_{c}_{count_str}.png'

            # print(c, count-1)
        return None


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # test zip path
    path = '../data/by_class.zip'

    # Test non-zip path
    path = '../data'

    seen = set()
    with data_reader(path) as z:
        for class_hex, file in z.relevant_file_walk():
            if class_hex not in seen:
                plt.imshow(file, interpolation='nearest')
                plt.show()

                seen.add(class_hex)
