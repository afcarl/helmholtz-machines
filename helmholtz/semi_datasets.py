
from __future__ import division

import numpy as np

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten, SourcewiseTransformer
from fuel.datasets.mnist import MNIST

from fuel_extras.semisupervised import mnist_subset, SemisupervisedDataStream
from fuel_extras.onehot import OneHot

supported_datasets = ['mnist', 'lbmnist']

def get_streams(data_name, n_labeled, batch_size, small_batch_size=None):
    # Our usual train/valid/test data streams...
    if small_batch_size is None:
        small_batch_size = max(1, batch_size // 10)

    if data_name == "mnist":
        from fuel.datasets.mnist import MNIST

        batch_size = (batch_size // 2, batch_size // 2)

        data_train = MNIST(which_sets=('train',), subset=slice(0, 50000))
        data_valid = MNIST(which_sets=('train',), subset=slice(50000, 60000))
        data_test = MNIST(which_sets=('test',))

        train_labeled, train_unlabeled = mnist_subset(data_train, n_labeled)
        valid_labeled, valid_unlabeled = mnist_subset(data_valid, n_labeled)
        test_labeled, test_unlabeled = mnist_subset(data_test, n_labeled)

        train_stream = OneHot(
                            SemisupervisedDataStream(
                                mnist_subset(data_train, n_labeled),
                                batch_size=batch_size),
                            which_sources='targets', n_labels=10)

        valid_stream = OneHot(
                            SemisupervisedDataStream(
                                mnist_subset(data_valid, data_valid.num_examples),
                                batch_size=(2*batch_size[0], 0)),
                            which_sources='targets', n_labels=10)

        test_stream = OneHot(
                            SemisupervisedDataStream(
                                mnist_subset(data_test, data_test.num_examples),
                                batch_size=(2*batch_size[0], 0)),
                            which_sources='targets', n_labels=10)


        return 28*28, 10, train_stream, valid_stream, test_stream
    elif data_name == "lbmnist":
        from fuel.datasets.hdf5 import H5PYDataset

        fname = "data/" + data_name + ".hdf5"
        batch_size = (batch_size // 2, batch_size // 2)

        data_train = H5PYDataset(fname, which_sets=["train"], sources=['features', 'targets'], load_in_memory=True)
        data_valid = H5PYDataset(fname, which_sets=["valid"], sources=['features', 'targets'], load_in_memory=True)
        data_test = H5PYDataset(fname, which_sets=["test"], sources=['features', 'targets'], load_in_memory=True)

        train_stream = SemisupervisedDataStream(
                            mnist_subset(data_train, n_labeled),
                            batch_size=batch_size)

        valid_stream = SemisupervisedDataStream(
                            mnist_subset(data_valid, n_labeled),
                            batch_size=batch_size)

        test_stream = SemisupervisedDataStream(
                            mnist_subset(data_test, n_labeled),
                            batch_size=batch_size)

        return 28*28, 10, train_stream, valid_stream, test_stream
    else:
        raise ValueError
