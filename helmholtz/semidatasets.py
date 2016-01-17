
from __future__ import division

import numpy as np

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten, SourcewiseTransformer

supported_datasets = ["mnist", "lbmnist"]


def get_streams(data_name, batch_size, small_batch_size=None):
    """ Returns	x_dim, y_dim, data_train, data_valid, data_test
    """
    if small_batch_size is None:
        small_batch_size = max(1, batch_size // 10)

    raise NotImplemented


def get_data(data_name):
    """ Return x_dim, y_dim, train_data, valid_data, test_data """

    if data_name == "mnist":
        from fuel.datasets.mnist import MNIST

        data_train = H5PYDataset(fname, which_sets=["train"], sources=['features', 'targets'], load_in_memory=True)
        data_valid = H5PYDataset(fname, which_sets=["test"], sources=['features', 'targets'], load_in_memory=True)
        data_test  = H5PYDataset(fname, which_sets=["test"], sources=['features', 'targets'], load_in_memory=True)

        some_features, some_targets = data_train.get_data(None, slice(0, 100))
        assert some_features.shape[0] == 100
        assert some_targets.shape[0] == 100

        x_dim = np.prod(some_features.shape)
        y_dim = np.prod(some_targets.shape)

        return x_dim, y_dim, data_train, data_valid, data_test

    elif data_name == "lbmnist":
        from fuel.datasets.hdf5 import H5PYDataset

        fname = "data/lbmnist.hdf5"

        data_train = H5PYDataset(fname, which_sets=["train"], sources=['features', 'targets'], load_in_memory=True)
        data_valid = H5PYDataset(fname, which_sets=["valid"], sources=['features', 'targets'], load_in_memory=True)
        data_test  = H5PYDataset(fname, which_sets=["test"], sources=['features', 'targets'], load_in_memory=True)

        some_features, some_targets = data_train.get_data(None, slice(0, 100))
        assert some_features.shape[0] == 100
        assert some_targets.shape[0] == 100

        x_dim = np.prod(some_features.shape)
        y_dim = np.prod(some_targets.shape)

        return x_dim, y_dim, data_train, data_valid, data_test
    else:
        raise ValueError("Unknown dataset %s" % data_name)
