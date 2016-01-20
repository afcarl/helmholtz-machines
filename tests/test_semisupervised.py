
import unittest

from fuel_extras.semisupervised import *

def test_shape():
    labeled, unlabeled = mnist_subset(1000)
    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(50, 50))
    it = stream.get_epoch_iterator(as_dict=True)
    batch = next(it)

    print(batch.keys)

    for key, val in batch.items():
        assert val.shape[0] == 100

def test_count():
    labeled, unlabeled = mnist_subset(1000)
    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(50, 50))

    total_batches = 0
    for i in enumerate(stream.get_epoch_iterator()):
        total_batches = total_batches + 1

    assert total_batches == ((60000 - 1000) / 50)
