
import numpy
import theano
import unittest

from blocks.bricks import MLP, Logistic
from blocks.initialization import IsotropicGaussian, Constant

from helmholtz.prob_layers import *

floatX = theano.config.floatX


inits = {
    'weights_init': IsotropicGaussian(0.1),
    'biases_init': Constant(-1.0),
}

#---------------------------------------------------------------------------


def test_benoulli_top_layer():
    # Setup layer
    dim_x = 100

    l = BernoulliTopLayer(dim_x, name="layer", **inits)
    l.initialize()

    n_samples = tensor.iscalar('n_samples')
    x_expected = l.sample_expected()
    x, x_log_prob = l.sample(n_samples)

    do = theano.function([n_samples],
                         [x_expected, x, x_log_prob],
                         allow_input_downcast=True)

    x_expected, x, x_log_prob = do(10)

    assert x_expected.shape == (dim_x,)
    assert x.shape == (10, dim_x)
    assert x_log_prob.shape == (10,)


def test_benoulli_layer():
    # Setup layer
    dim_y = 50
    dim_x = 100

    mlp = MLP([Logistic()], [dim_y, dim_x], **inits)

    l = BernoulliLayer(mlp, name="layer", **inits)
    l.initialize()

    y = tensor.fmatrix('y')
    x_expected = l.sample_expected(y)
    x, x_log_prob = l.sample(y)

    do = theano.function([y],
                         [x_expected, x, x_log_prob],
                         allow_input_downcast=True)

    y = numpy.eye(50, dtype=numpy.float32)

    x_expected, x, x_log_prob = do(y)

    assert x_expected.shape == (50, dim_x)
    assert x.shape == (50, dim_x)
    assert x_log_prob.shape == (50,)
