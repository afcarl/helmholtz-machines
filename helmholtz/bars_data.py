#!/usr/bin/env python

from __future__ import division, print_function

import numpy

from collections import OrderedDict

from fuel.datasets import IndexableDataset


class Bars(IndexableDataset):
    provides_sources = ('features', 'latents')

    def __init__(self, num_examples=1000, width=4, sparsity=2, **kwargs):
        # Create dataset
        H = 2 * width
        U = numpy.random.uniform(size=(num_examples, H), low=0, high=1)
        s = U < (sparsity / H)

        features = numpy.zeros((num_examples, width, width))
        for h in xrange(width):
            features[s[:,h      ], h, :] = 1.
            features[s[:,h+width], :, h] = 1.
        features = features.reshape((num_examples, width**2))

        latents = s.astype('float32')
        features = features.astype('float32')

        data = OrderedDict([
            ('features', features),
            ('latents', latents),
        ])

        super(Bars, self).__init__(data, **kwargs)


if __name__ == "__main__":
    import pylab
    
    width = 4
    
    data = Bars(num_examples=100, width=width)
    features, latents = data.get_data(None, slice(0, 100))
    
    for i in xrange(100):
        pylab.subplot(10, 10, i+1)
        pylab.imshow(features[i].reshape([width, width]), interpolation="nearest")
        pylab.axis("off")
    pylab.show(block=True)

