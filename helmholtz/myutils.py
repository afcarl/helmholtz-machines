
from __future__ import division

from abc import ABCMeta, abstractmethod

import ipdb
import numpy
import six
import theano

from collections import OrderedDict

from theano import tensor
from blocks.initialization import NdarrayInitialization, Uniform


def merge_gradients(old_gradients, new_gradients, scale=1.):
    """Take and merge multiple ordered dicts 
    """
    if isinstance(new_gradients, (dict, OrderedDict)):
        new_gradients = [new_gradients]

    for gradients in new_gradients:
        assert isinstance(gradients, (dict, OrderedDict))
        for key, val in gradients.items():
            if old_gradients.has_key(key):
                old_gradients[key] = old_gradients[key] + scale * val
            else:       
                old_gradients[key] = scale * val
    return old_gradients

#-----------------------------------------------------------------------------

class ShapeDependentInitialization(NdarrayInitialization):
    """Initialize 

    Parameters
    ----------
    weights_init : :class:`NdarrayInitialization` instance
        The unscaled initialization scheme to initialize the weights with.
    """
    def __init__(self, weights_init):
        super(ShapeDependentInitialization, self).__init__()
        self.weights_init = weights_init

    def generate(self, rng, shape):
        weights = self.weights_init.generate(rng, shape)
        scale = self.scale_func(*shape)
        return scale*weights

    # TODO: Abstract
    def scale_func(self, *shape):
        pass


class TanhInitialization(ShapeDependentInitialization):
    """Normalized initialization for tanh MLPs. 

    This class initializes parameters by drawing from the uniform 
    distribution   with the interval 

        [- sqrt(6)/sqrt(dim_in+dim_out)  .. sqrt(6)/sqrt(dim_in+dim_out)]
    """
    def __init__(self):
        super(TanhInitialization, self).__init__(Uniform(mean=0., width=2.))

    def scale_func(self, dim_in, dim_out):
        return numpy.sqrt(6)/numpy.sqrt(dim_in+dim_out)


class RWSInitialization(ShapeDependentInitialization):
    def __init__(self, factor=1.):
        super(RWSInitialization, self).__init__(Uniform(mean=0., width=2.))
        self.factor = factor

    def scale_func(self, dim_in, dim_out):
        return self.factor * numpy.sqrt(6)/numpy.sqrt(dim_in+dim_out)/dim_in

#-----------------------------------------------------------------------------

class GradientMonitor(object):
    def __init__(self, gradients, prefix=""):
        self.gradients = gradients
        self.prefix = prefix
        pass

    def vars(self):
        prefix = self.prefix 
        monitor_vars = []

        aggregators = {
            'min':   tensor.min,
            'max':   tensor.max,
            'mean':  tensor.mean,
        }

        for key, value in six.iteritems(self.gradients):
            min = tensor.min(value)
            ipdb.set_trace()
            monitor_vars.append(monitor_vars)

        return monitor_vars


