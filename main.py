import os
from siamese_nn import SiameseNN
import cv2
import numpy as np
from sklearn.utils import shuffle


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

    shuffle(X_train, y_train)
    shuffle(X_test, y_test)

    X_train = [X_train[:, 0, :, :], X_train[:, 1, :, :]]
    X_test = [X_test[:, 0, :, :], X_test[:, 1, :, :]]
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = SiameseNN(learning_rate=0.01, batch_size=128)
    model.fit(X_train, y_train)


if __name__ == '__main__':
    main()
