
from __future__ import division

import numpy as np

from collections import OrderedDict

from fuel.datasets.base import IndexableDataset
from fuel.datasets.mnist import MNIST
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten, SourcewiseTransformer

from fuel_extras.semisupervised import mnist_subset, SemisupervisedDataStream
from fuel_extras.onehot import OneHot

supported_datasets = ['mnist', 'lbmnist']


def get_streams(data_name, n_labeled, batch_size, small_batch_size=None):
    # Our usual train/valid/test data streams...
    if small_batch_size is None:
        small_batch_size = max(1, batch_size // 10)

    if data_name == "mnist":
        from fuel.datasets.mnist import MNIST

        if small_batch_size is None:
            small_batch_size = batch_size
        batch_size = (batch_size // 2, batch_size // 2)

        data_train = MNIST(which_sets=('train',), subset=slice(0, 50000))
        train_labeled, train_unlabeled = mnist_subset(data_train, n_labeled)

        # Valid data
        data_valid = MNIST(which_sets=('train',), subset=slice(50000, 60000))
        state = data_valid.open()
        features, labels = data_valid.get_data(state, slice(0, data_valid.num_examples))
        data_valid.close(state)
        features = np.cast[np.float32](features / features.max())
        valid_data = OrderedDict([
            ('features', features),
            ('targets', labels),
            ('mask', np.ones((data_valid.num_examples, 1), dtype='int32')),
        ])
        data_valid = IndexableDataset(valid_data)

        train_stream = Flatten(
            OneHot(
                SemisupervisedDataStream(
                    mnist_subset(data_train, n_labeled),
                    batch_size=batch_size),
                which_sources='targets', n_labels=10),
            which_sources='features')

        valid_stream = Flatten(
            OneHot(
                DataStream(
                    data_valid, 
                    iteration_scheme=ShuffledScheme(data_valid.num_examples, small_batch_size)),
                which_sources='targets', n_labels=10),
            which_sources='features')
        del state, features, labels, valid_data, data_valid

        # Test data
        data_test = MNIST(which_sets=('test',))
        state = data_test.open()
        features, labels = data_test.get_data(state, slice(0, data_test.num_examples))
        features = np.cast[np.float32](features / features.max())
        data_test.close(state)
        test_data = OrderedDict([
            ('features', features),
            ('targets', labels),
            ('mask', np.ones((data_test.num_examples, 1), dtype='int32')),
        ])
        data_test = IndexableDataset(test_data)

        test_stream = Flatten(
            OneHot(
                DataStream(
                    data_test, 
                    iteration_scheme=ShuffledScheme(data_test.num_examples, small_batch_size)),
                which_sources='targets', n_labels=10),
            which_sources='features')

        del state, features, labels, test_data, data_test

        return 28 * 28, 10, train_stream, valid_stream, test_stream
    elif data_name == "lbmnist":
        raise ValueError
    else:
        raise ValueError
