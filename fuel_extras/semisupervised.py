#!/usr/bin/env python

from __future__ import division, print_function

import six
import ipdb
import numpy

from collections import Iterable

from picklable_itertools import cycle
from picklable_itertools.extras import partition_all

from fuel.streams import DataStream
from fuel.streams import AbstractDataStream

def ceildiv(a, b):
    return -(-a // b)


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
