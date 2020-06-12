import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import os
from .utils import load_image, equalize


def get_data(target_size=512, test_size=0.25, random_state=23,
             preprocessing=True, label='senior', label_type='region'):
    """

    :param target_size:
    :param test_size:
    :param random_state:
    :param preprocessing:
    :param label:
    :param label_type:
    :return:
    """
    assert label in ['senior', 'junior'], print("label field must be either 'senior' or 'junior'.")
    assert label_type in ['global', 'region'], print("label_type must be either 'global' or 'region'.")

    # load annotations from csv
    ds = pd.read_csv('../data/public-annotations.csv')

    X = []
    y = []
    for it in tqdm(ds.itertuples()):
        im = load_image(os.path.join('../data/public-cohen-subset/',
                                    it.filename), target_size)
        if label == 'senior' and label_type == 'region':
            bs = np.reshape(it[2:8], (2,3)).T
        if label == 'junior' and label_type == 'region':
            bs = np.reshape(it[9:15], (2,3)).T
        if label == 'senior' and label_type == 'global':
            bs = it[8]
        if label == 'junior' and label_type == 'global':
            bs = it[15]

        if preprocessing:
            im = equalize(im).numpy()

        X.append(im)
        y.append(bs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
