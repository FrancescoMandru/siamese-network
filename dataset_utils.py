import tensorflow as tf
import numpy as np
import os
import math
from scipy.spatial import distance
import random
import cv2


def random_class_ids(lower, upper):
    """
    Get two random integers that are not equal.
    Note: In some cases (such as there being only one sample of a class) there may be an endless loop here. This
    will only happen on fairly exotic datasets though. May have to address in future.
    :param lower: Lower limit inclusive of the random integer.
    :param upper: Upper limit inclusive of the random integer. Need to use -1 for random indices.
    :return: Tuple of (integer, integer)
    """
    int_1 = random.randint(lower, upper)
    int_2 = random.randint(lower, upper)
    while int_1 == int_2:
      int_1 = random.randint(lower, upper)
      int_2 = random.randint(lower, upper)
    return int_1, int_2


def custom_sort(x):
    """
    Custom sort function used to build positive pairs properly
    in create_positive_pairs function
    """
    return(x[-10:-8])


def create_positive_pairs(mother_dir):
    """ Function which create positive pairs according with the following rule:
    For each category directory take two images at couple of two since in the directory
    each product is followed by its related product.
    Input:
    Mother directory with all the category directories
    Ouput:
    List of couples
    List of labels
    """
    positive_pairs = []
    positive_labels = []

    exclude_directories = set(['general'])
    for subdir, dirs, files in os.walk(mother_dir):
        category_pairs = []
        dirs[:] = [d for d in dirs if d not in exclude_directories] # exclude directory if in exclude list
        for file in files:
            if not file.startswith('.'):
                category_pairs.append(os.path.join(subdir, file))

        sorted_files = sorted(category_pairs, key = custom_sort)

        for one, two in zip(sorted_files[0::2], sorted_files[1::2]):
            positive_pairs.append([one, two])
            positive_labels.append(1.0)

    return positive_pairs, positive_labels


def compute_similarity(img_one,img_two):
    """Performs image resizing just to compute the
    cosine similarity faster
    Input:
      Two images
    Output:
      Cosine Similarity
    """

    x = cv2.resize(img_one, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
    y = cv2.resize(img_two, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)

    x = x.ravel().reshape(-1, 1)
    y = y.ravel().reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        dist = 0
    else:
        dist = 1 - distance.cosine(x, y)
    return dist


def create_negative_pairs(mother_dir, num_negative_pairs):
    """ Build negative pairs online
    Input:
    mother_dir: Mother directory containing all the directories with all the categories
    num_negative_pairs: Number of negatives you want to create online
    Output:
    List of couples
    List of labels
    """
    negative_pairs = []
    negative_labels = []

    dirlist = [item for item in os.listdir(mother_dir) if os.path.isdir(os.path.join(mother_dir, item))]
    count = 0
    while count < num_negative_pairs:
        rnd_class_one, rnd_class_two = random_class_ids(0,len(dirlist)-1)
        rnd_class_one = os.path.join(mother_dir, dirlist[rnd_class_one])
        rnd_class_two = os.path.join(mother_dir, dirlist[rnd_class_two])

        a = [x for x in os.listdir(rnd_class_one) if not x.startswith('.')]
        rnd_elm_one = random.choice(a)
        one = os.path.join(rnd_class_one, rnd_elm_one)

        a = [x for x in os.listdir(rnd_class_two) if not x.startswith('.')]
        rnd_elm_two = random.choice(a)
        two = os.path.join(rnd_class_two, rnd_elm_two)

        negative_pairs.append([one, two])
        negative_labels.append(0.0)
        count += 1

    return negative_pairs, negative_labels


def create_pairs(mother_dir,negative_factor):
    """Main function to create the whole dataset
    Input:
    mother_dir: Mother directory containing all the directories with all the categories
    negative_factor: Integer to multiply with the number of positive pairs to build imbalanced dataset
    Output:
    Data and labels
    """
    positive_pairs, positive_labels = create_positive_pairs(mother_dir)
    print("POSITIVES CREATED: ", len(positive_pairs))
    num_negative_pairs = len(positive_pairs) * negative_factor
    negative_pairs, negative_labels = create_negative_pairs(mother_dir, num_negative_pairs)
    print("NEGATIVES CREATED: ", len(negative_pairs))

    return np.array(positive_pairs + negative_pairs), np.array(positive_labels + negative_labels)


class PairDataGenerator(tf.keras.utils.Sequence):

    """ Generator of dataset. This generator allows you to avoid congestion of the RAM of the GPU
    working one batch at time in RAM """

    def __init__(self, files, labels, img_height=224, img_width=224, img_channel=3, batch_size=64):
        self.files = files
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.batch_size = batch_size
        self.indices = np.arange(self.files.shape[0])


    def __len__(self) :
        return math.floor(self.files.shape[0] / self.batch_size)

    # Perform data augmentation
    def __augmentation__(self, img_one, img_two):

        rnd_num = random.uniform(0,1)
        if rnd_num <= 0.20:
            img_one = tf.image.flip_left_right(img_one)
        elif 0.2 < rnd_num <= 0.35:
            img_one = tf.keras.preprocessing.image.random_rotation(img_one,30)

        rnd_num = random.uniform(0,1)
        if rnd_num <= 0.20:
            img_two = tf.image.flip_left_right(img_two)
        elif 0.2 < rnd_num <= 0.35:
            img_two = tf.keras.preprocessing.image.random_rotation(img_two,30)

        return img_one, img_two

    # Get current batch based on idx
    def __getitem__(self, idx):

        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.files[inds]
        batch_lab = self.labels[inds]

        train_image_one = []
        train_image_two = []
        train_label = []

        img_resize = self.img_height


        for i in range(len(batch)):

            img_one = tf.io.read_file(batch[i][0])
            img_one = tf.image.convert_image_dtype( tf.io.decode_png(img_one, channels=self.img_channel), dtype='float32')
            img_one = tf.image.resize(img_one, (img_resize,img_resize))

            img_two = tf.io.read_file(batch[i][1])
            img_two = tf.image.convert_image_dtype( tf.io.decode_png(img_two, channels=self.img_channel), dtype='float32')
            img_two = tf.image.resize(img_two, (img_resize,img_resize))

            img_one, img_two = self.__augmentation__(img_one, img_two)

            train_image_one.append(img_one)
            train_image_two.append(img_two)

            train_label.append(batch_lab[i])

        train_image = [tf.convert_to_tensor(train_image_one),
                       tf.convert_to_tensor(train_image_two)]


        return train_image, tf.convert_to_tensor(train_label)

    # Shuffle avery epoch
    def on_epoch_end(self):
        np.random.shuffle(self.indices)