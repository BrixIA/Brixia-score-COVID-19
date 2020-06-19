import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from .utils import load_image, equalize
l_path = Path(__file__).parent


def get_data(target_size=512, test_size=0.25, random_state=23,
             preprocessing=True, label='senior', label_type='region'):
    """
    Get train and test data from cohen dataset
    :param target_size: image dimension
    :param test_size: train test split (default 0.25)
    :param random_state: random seed
    :param preprocessing: whether to apply preprocessing
    :param label: gt label ['senior', 'junior']
    :param label_type:
    :return: 3x2 region score or global one ['global', 'region']
    """
    assert label in ['senior', 'junior'], print("label field must be either 'senior' or 'junior'.")
    assert label_type in ['global', 'region'], print("label_type must be either 'global' or 'region'.")

    # load annotations from csv
    ds = pd.read_csv((l_path / '../../data/public-annotations.csv').resolve())

    X = []
    y = []
    for it in tqdm(ds.itertuples()):
        im = load_image((l_path / '../../data/public-cohen-subset' / it.filename).resolve().as_posix(),
                        target_size)
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
