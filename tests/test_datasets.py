
import unittest

from nose.plugins.skip import Skip, SkipTest

import helmholtz.datasets as datasets



def test_shape():
    def check_dataset(name):
        try:
            x_dim, data_train, data_valid, data_test = datasets.get_data(name)
        except IOError as e:
            raise SkipTest

        for ds in (data_train, data_valid, data_test):
            features, = ds.get_data(None, slice(0, 100))

            features = features.reshape([100, -1])
            assert features.shape == (100, x_dim)

    for name in datasets.supported_datasets:
        yield check_dataset, name

def test_range():
    def check_dataset(name):
        try:
            x_dim, data_train, data_valid, data_test = datasets.get_data(name)
        except IOError as e:
            raise SkipTest

        for ds in (data_train, data_valid, data_test):
            features, = ds.get_data(None, slice(0, 100))

            features = features.reshape([100, -1])
            assert (features >= 0).all()
            assert (features <= 1).all()

    for name in datasets.supported_datasets:
        yield check_dataset, name
