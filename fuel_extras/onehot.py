from __future__ import division

import numpy

from fuel.transformers import SourcewiseTransformer

class OneHot(SourcewiseTransformer):
    def __init__(self, data_stream, n_labels, **kwargs):
        super(OneHot, self).__init__(data_stream, False, **kwargs)

        self.n_labels = n_labels

    #def transform_source_example(self, source_example, _):
    #    return numpy.asarray(source_example).flatten()

    def transform_source_batch(self, source_batch, _):

        batch_size = source_batch.shape[0]

        one_hot = numpy.zeros((batch_size, self.n_labels), dtype=numpy.float32)
        one_hot[numpy.arange(batch_size),source_batch[:,0]] = 1
        return one_hot
