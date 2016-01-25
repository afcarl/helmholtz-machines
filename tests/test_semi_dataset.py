
import unittest

from helmholtz.semi_datasets import *

def test_mnist():
    x_dim, y_dim, train_stream, valid_stream, test_stream = get_streams('mnist', 100, 100)

    it = train_stream.get_epoch_iterator()
    features, targets, mask = next(it)

    print(features.shape)
    print(targets.shape)
    print(mask.shape)


    assert features.shape == (100, 28*28)
    assert targets.shape == (100, 10)
    assert mask.shape == (100, 1)

    assert (targets.sum(axis=1) == 1).all()

    # MNIST validation set
    num_batches = 0
    for features, targets, mask in valid_stream.get_epoch_iterator():
        assert features.shape == (100, 28*28)
        assert targets.shape == (100, 10)
        assert mask.shape == (100, 1)

        num_batches = num_batches + 1

    import ipdb; ipdb.set_trace()


#def test_lbmnist():
#    x_dim, y_dim, train_stream, valid_stream, test_stream = get_streams('lbmnist', 100, 100)
#
#    it = train_stream.get_epoch_iterator()
#    features, targets, mask = next(it)
#
#    print features.shape
#    print targets.shape
#    print mask.shape
