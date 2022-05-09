import os
from siamese_nn import SiameseNN
import cv2
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
import logging as log
import sys
import json

logfile_name = f"run_{str(datetime.now()).replace(' ', '_')}"
log.basicConfig(filename=f"logs/{logfile_name}.log", format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', level=log.INFO)
log.getLogger().addHandler(log.StreamHandler(sys.stdout))


def extract_images(file, all_img_dir):
    X = []
    y = []
    with open(file) as f:
        imgs = [l.rstrip('\n').split('\t') for l in f.readlines()][1:]
        for img in imgs:
            person1 = img[0]
            i1 = f"{person1}_{int(img[1]):04d}.jpg"
            im1 = cv2.imread(os.path.join(all_img_dir, person1, i1), cv2.IMREAD_GRAYSCALE)
            # Case where both images are the same person:
            if len(img) == 3:
                i2 = f"{person1}_{int(img[2]):04d}.jpg"
                im2 = cv2.imread(os.path.join(all_img_dir, person1, i2), cv2.IMREAD_GRAYSCALE)
                y.append(1)
            # Case where both images are NOT the same person:
            elif len(img) == 4:
                person2 = img[2]
                i2 = f"{person2}_{int(img[3]):04d}.jpg"
                im2 = cv2.imread(os.path.join(all_img_dir, person2, i2), cv2.IMREAD_GRAYSCALE)
                y.append(0)
            else:
                raise ValueError(f"Input file does not accept row with length of {len(img)}.")
            X.append([im1/255, im2/255])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y


def load_data():
    train_file = 'data/pairsDevTrain.txt'
    test_file = 'data/pairsDevTest.txt'

    all_img_dir = 'data/lfwa'
    X_train, y_train = extract_images(train_file, all_img_dir)
    X_test, y_test = extract_images(test_file, all_img_dir)

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    X_train = [X_train[:, 0, :, :], X_train[:, 1, :, :]]
    X_test = [X_test[:, 0, :, :], X_test[:, 1, :, :]]
    return X_train, y_train, X_test, y_test


def run_model(X_train, y_train, X_test, y_test, params=None):
    """
    :param X_train: Train samples. size - (2, m, 250, 250)
    :param y_train: Train labels. size - (m, 1)
    :param X_test: Test samples. size - (2, m, 250, 250)
    :param y_test: Test labels. size - (m, 1)
    :param params: Dictionary of params for the model -
                   example: {'learning_rate': 1e-2, 'l2_regularization': 1e-3, 'batch_size': 128}
    """
    if params is None:
        params = {}
    run_name = f"{json.dumps(params)}_{str(datetime.now()).replace(' ', '-')}"
    model = SiameseNN(**params, name=run_name)
    model.fit(X_train, y_train, epochs=1)
    log.info(f"Run name - {run_name}\ntrain accuracy - {model.score(X_train, y_train)} test accuracy - {model.score(X_test, y_test)}\n")


def main():
    X_train, y_train, X_test, y_test = load_data()

    run_model(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
