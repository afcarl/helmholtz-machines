
import unittest

from nose.plugins.skip import Skip, SkipTest

import helmholtz.datasets as datasets



def test_shape():
    def check_dataset(name):
        try:
            x_dim, data_train, data_valid, data_test = datasets.get_streams(name, batch_size=10, small_batch_size=10)
        except IOError as e:
            raise SkipTest

        for ds in (data_train, data_valid, data_test):
            features, = next(ds.get_epoch_iterator())

            features = features.reshape([10, -1])
            assert features.shape == (10, x_dim)

    for name in datasets.supported_datasets:
        yield check_dataset, name


def test_range():
    def check_dataset(name):
        try:
            x_dim, data_train, data_valid, data_test = datasets.get_streams(name, batch_size=10, small_batch_size=10)
        except IOError as e:
            raise SkipTest

        for ds in (data_train, data_valid, data_test):
            features, = next(ds.get_epoch_iterator())

            features = features.reshape([10, -1])
            assert (features >= 0).all()
            assert (features <= 1).all()

    for name in datasets.supported_datasets:
        yield check_dataset, name
