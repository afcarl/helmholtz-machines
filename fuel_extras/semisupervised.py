#!/usr/bin/env python

from __future__ import division, print_function

import six
import ipdb
import numpy

from collections import OrderedDict, Iterable
from picklable_itertools import chain, cycle, imap
from picklable_itertools.extras import partition_all
from six.moves import xrange

from fuel.streams import DataStream
from fuel.datasets.base import IndexableDataset
from fuel.datasets.mnist import MNIST
from fuel.streams import AbstractDataStream

def ceildiv(a, b):
    return -(-a // b)

def mnist_subset(dataset, n_labeled):
    """Split a dataset into an labeled and an unlabeled subset.

    Parameters
    ----------
    dataset : DataSet
    n_labeled : int
        size of labeled subset

    Returns
    -------
    (labeled, unlabeled)
        Two subsets. 
    """
    state = dataset.open()

    n_labaled_per_class = n_labeled // 10
    n_total = dataset.num_examples

    features, labels = dataset.get_data(state, slice(0, n_total))
    dataset.close(state)

    features = numpy.cast[numpy.float32](features / features.max())
    #labels = numpy.cast[numpy.float32](labels)

    labeled_idx = []
    unlabeled_idx = []
    for c in xrange(10):
        cidx = numpy.where(labels == c)[0]
        labeled_idx.append(cidx[:n_labaled_per_class])
        unlabeled_idx.append(cidx[n_labaled_per_class:])

    labeled_idx = numpy.sort(numpy.concatenate(labeled_idx))
    unlabeled_idx = numpy.sort(numpy.concatenate(unlabeled_idx))

    unlabeled_data = OrderedDict([
        ('features', features[unlabeled_idx]),
        ('targets', labels[unlabeled_idx]),
        ('mask', numpy.zeros((len(unlabeled_idx), 1), dtype='int32')),
    ])

    labeled_data = OrderedDict([
        ('features', features[labeled_idx]),
        ('targets', labels[labeled_idx]),
        ('mask', numpy.ones((len(labeled_idx), 1), dtype='int32')),
    ])

    labeled = IndexableDataset(labeled_data)
    unlabeled = IndexableDataset(unlabeled_data)

    return labeled, unlabeled


class SemisupervisedDataIterator(six.Iterator):
    def __init__(self, data_stream, batch_size, as_dict=False):
        """
        Parameters
        ----------
        data_stream : SemisupervisedDataStream
        batch_size : int or tuple of int
        as_dict : bool
        """
        assert isinstance(data_stream, SemisupervisedDataStream)

        if not isinstance(batch_size, Iterable):
            batch_size = [batch_size // len(data_stream.datasets)] * len(data_stream.datasets)

        num_examples = [ds.num_examples for ds in data_stream.datasets]
        num_batches = [ceildiv(ne, bs) for ne, bs in zip(num_examples, batch_size)]
        max_examples = max(num_examples)
        max_batches = max(num_batches)

        indices = [range(ne) for ne in num_examples]
        iterators = [cycle(partition_all(bs, idx)) for idx, bs in zip(indices, batch_size)]

        self.data_stream = data_stream
        self.batch_size = batch_size
        self.as_dict = as_dict
        self.iterators = iterators
        self.remaining_batches = max_batches

    def __iter__(self):
        return self

    def __next__(self):
        data_stream = self.data_stream

        if self.remaining_batches <= 0:
            raise StopIteration
        self.remaining_batches = self.remaining_batches - 1

        data = None
        for ds, ds_state, it in zip(data_stream.datasets, data_stream.ds_states, self.iterators):
            more_data = ds.get_data(ds_state, next(it))

            # Merge data into commen dictionary
            if data:
                data = [numpy.concatenate((d, md)) for d, md in zip(data, more_data)]
            else:
                data = more_data

        if self.as_dict:
            return dict(zip(data_stream.sources, data))
        else:
            return data


class SemisupervisedDataStream(AbstractDataStream):
    """A DataStream that merges two datasets.
    """
    def __init__(self, datasets, batch_size):
        """
        Parameters
        ----------
        datasetd : tuple of DataSet
        """
        self.datasets = datasets
        self.batch_size = batch_size

        # Open all datasets
        self.ds_states = [ds.open() for ds in self.datasets]

    @property
    def produces_examples(self):
        return False

    @property
    def sources(self):
        return self.datasets[0].sources

    def reset(self):
        self.ds_states = [ds.reset(state) for ds, state in zip(self.datasets, self.ds_states)]
        self._fresh_state = True

    def close(self):
        self.ds_states = [ds.close(state) for ds, state in zip(self.datasets, self.ds_states)]

    def next_epoch(self):
        self.ds_states = [ds.next_epoch(state) for ds, state in zip(self.datasets, self.ds_states)]

    def get_data(self, request=None):
        raise NotImplemented

    def get_epoch_iterator(self, as_dict=False):
        return SemisupervisedDataIterator(self, batch_size=self.batch_size, as_dict=as_dict)

#----------------------------------------------------------------------------


if __name__ == "__main__":
    mnist = MNIST()
    labeled, unlabeled = mnist_subset(mnist, 1000)

    print("Size of labeled subset:   %d" % labeled.num_examples)
    print("Size of unlabeled subset: %d" % unlabeled.num_examples)

    stream = SemisupervisedDataStream(datasets=(labeled, unlabeled), batch_size=(50, 50))

    for i, batch in enumerate(stream.get_epoch_iterator(as_dict=True)):
        print(i)
