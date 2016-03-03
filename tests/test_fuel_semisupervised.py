
from __future__ import division, print_function

import unittest

from fuel.datasets.mnist import MNIST

from helmholtz.semi_datasets import mnist_subset
from fuel_extras.semisupervised import *


def test_shape():
    dataset = MNIST(which_sets=('train',))
    labeled, unlabeled = mnist_subset(dataset, 1000)
    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(50, 50))
    it = stream.get_epoch_iterator(as_dict=True)
    batch = next(it)

    print(batch.keys())

    for key, val in batch.items():
        assert val.shape[0] == 100
        print("%s: %s" % (key, val.shape))


def test_mask():
    dataset = MNIST(which_sets=('train',))
    labeled, unlabeled = mnist_subset(dataset, 1000)
    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(90, 10))

    batch = next(stream.get_epoch_iterator(as_dict=True))
    assert batch['mask'].sum() == 90

    for batch in stream.get_epoch_iterator(as_dict=True):
        if batch['mask'].sum() != 90:
            import ipdb; ipdb.set_trace()
        

def test_count():
    dataset = MNIST(which_sets=('train',))
    labeled, unlabeled = mnist_subset(dataset, 1000)
    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(50, 50))

    total_batches = 0
    for i in enumerate(stream.get_epoch_iterator()):
        total_batches = total_batches + 1

    assert total_batches == ((60000 - 1000) / 50)
