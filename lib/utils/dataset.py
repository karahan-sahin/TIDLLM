"""
This file contains the functions to get the dataset

Functions:
    get_dataset: Get the dataset from the dataset folder

"""

import glob
from lib.data.dataset import PoseDistanceDataset
from sklearn.model_selection import train_test_split

def get_dataset(
    DATASET_PATH='dataset',
    DATASET_EXTENSION='.npy',
    DATASET_ENCODING='utf-8',
    DATA_DISTRIBUTION='80-20',
    DATASET_CONFIG=None,
    RANDOM_STATE=42
):
    """
    Get the dataset from the dataset folder
    """

    data = glob.glob(DATASET_PATH + f'*{DATASET_EXTENSION}')

    print('Data size:', len(data))

    # Check if the dataset is empty or sum is neq to 100
    if all([i.isdigit() for i in DATA_DISTRIBUTION.split('-')]) \
        and sum([int(i) for i in DATA_DISTRIBUTION.split('-')]) != 100:
        raise ValueError('DATA_DISTRIBUTION must be in the format "\d{2}-\d{2}" or "\d{2}-\d{2}-\d{2}"')

    if len(DATA_DISTRIBUTION.split('-')) == 2:
        train_dist, test_dist = DATA_DISTRIBUTION.split('-')
        train_dist, test_dist = int(train_dist) / 100, int(test_dist) / 100
        
        X_train, X_val = train_test_split(data, test_size=test_dist, random_state=RANDOM_STATE)
        train_dataset = PoseDistanceDataset(X_train, **DATASET_CONFIG)
        val_dataset = PoseDistanceDataset(X_val, **DATASET_CONFIG)

        return train_dataset, val_dataset
    
    elif len(DATA_DISTRIBUTION.split('-')) == 3:
        train_dist, test_dist, val_dist = DATA_DISTRIBUTION.split('-')
        train_dist, test_dist, val_dist = int(train_dist) / 100, int(test_dist) / 100, int(val_dist) / 100
        
        X_train, X_val = train_test_split(data, test_size=val_dist, random_state=RANDOM_STATE)
        X_train, X_test = train_test_split(X_train, test_size=test_dist, random_state=RANDOM_STATE)
        
        train_dataset = PoseDistanceDataset(X_train, **DATASET_CONFIG)
        test_dataset = PoseDistanceDataset(X_test, **DATASET_CONFIG)
        val_dataset = PoseDistanceDataset(X_val, **DATASET_CONFIG)

        return train_dataset, test_dataset, val_dataset
