
import unittest

from helmholtz.semi_datasets import *

def test_mnist():
    x_dim, y_dim, train_stream, valid_stream, test_stream = get_streams('mnist', 100, 100)


    #for features, targets, mask in train_stream.get_epoch_iterator():
    #    print features.shape
    #    print targets.shape
    #    print mask.shape

    it = train_stream.get_epoch_iterator()
    features, targets, mask = next(it)

    assert features.shape == (100, 1, 28, 28)
    assert targets.shape == (100, 10)
    assert mask.shape == (100, )

    assert (targets.sum(axis=1) == 1).all()

#def test_lbmnist():
#    x_dim, y_dim, train_stream, valid_stream, test_stream = get_streams('lbmnist', 100, 100)
#
#    it = train_stream.get_epoch_iterator()
#    features, targets, mask = next(it)
#
#    print features.shape
#    print targets.shape
#    print mask.shape
