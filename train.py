import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split

import tensorflow as tf

from dataset_utils import create_pairs, PairDataGenerator
from utils import compile_and_fit
from model import siameseLeg, siameseNet

if __name__ == '__main__':
    ROOT_DIR = 'PATH_OF_YOUR_DATASET'  # To Fix according to your path
    BATCH_SIZE = 128
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    # Multiplier of negative pairs w.r.t to the number of positive pairs
    # 3 means that we have double number of negative pairs
    neg_factor = 3
    X, y = create_pairs(ROOT_DIR, neg_factor)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.15, random_state=42)

    train_gen = PairDataGenerator(X_train, y_train, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE)
    val_gen = PairDataGenerator(X_val, y_val, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE)
    # test_gen = PairDataGenerator(X_test,y_test,IMG_HEIGHT,IMG_WIDTH,CHANNELS,1)

    print("\n-----------------------\n")
    print("TRAIN: ", X_train.shape)
    print("VAL: ", X_val.shape)
    # print("TEST: ", X_test.shape)
    print("\n-----------------------\n")
    tl, cl = np.unique(y_train, return_counts=True)
    print("TRAIN - LABEL: {} NUM: {} LABEL: {} NUM: {}".format(tl[0], cl[0], tl[1], cl[1]))
    # Deal with class imbalance according with Tensorflow documentation
    weight_for_0 = (1 / cl[0]) * (len(y_train)) / 2.0
    weight_for_1 = (1 / cl[1]) * (len(y_train)) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    # Initial network bias 
    initial_bias = np.log([cl[1] / cl[0]])
    print("Initial Bias: ".format(initial_bias[0]))

    tl, cl = np.unique(y_val, return_counts=True)
    print("VAL - LABEL: {} NUM: {} LABEL: {} NUM: {}".format(tl[0], cl[0], tl[1], cl[1]))
    # tl, cl = np.unique(y_test,return_counts=True)
    # print("TEST - LABEL: {} NUM: {} LABEL: {} NUM: {}".format(tl[0],cl[0],tl[1],cl[1]))

    siamese_model = siameseNet(siameseLeg,
                               (2, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS),
                               output_bias=initial_bias,
                               pretrained_leg=True)

    print("\n------> MODEL SUMMARY <------ \n")
    siamese_model.summary()

    # Tensorboard data folder
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    fit_logdir = "logs/fit/" + date

    # Compile the model and fit 
    compile_and_fit(siamese_model,
                    train_gen,
                    val_gen,
                    fit_logdir,
                    max_epochs=10)

    # Save model
    MODEL_DIR = './saved_model'
    version = datetime.today().strftime("%Y%m%d")
    export_path = os.path.join(MODEL_DIR, version)

    siamese_model.save(export_path)

    print('MODEL SAVED AT: {}\n'.format(export_path))
