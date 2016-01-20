
import numpy
import theano
import unittest

from blocks.bricks import MLP, Logistic
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.select import Selector
from blocks.roles import PARAMETER

from collections import OrderedDict

from helmholtz.nade import *

floatX = theano.config.floatX


inits = {
    'weights_init': IsotropicGaussian(0.1),
    'biases_init': Constant(-1.0),
}

#---------------------------------------------------------------------------

def test_nade_top_layer():
    # Setup layer
    dim_x = 100

    l = NADETopLayer(dim_x, name="layer", **inits)
    l.initialize()

    # Test sample
    n_samples = tensor.iscalar('n_samples')
    x, x_log_prob = l.sample(n_samples)

    do_sample = theano.function([n_samples],
                                [x, x_log_prob],
                                allow_input_downcast=True)

    a, a_log_prob = do_sample(10)

    #assert x_expected.shape == (dim_x,)
    assert a.shape == (10, dim_x)
    assert a_log_prob.shape == (10,)


    # Test log_prob
    x = tensor.fmatrix('x')
    x_log_prob = l.log_prob(x)

    do_prob = theano.function([x],
                              x_log_prob,
                              allow_input_downcast=True)

    b_log_prob = do_prob(a)

    assert b_log_prob.shape == (10,)
    assert numpy.allclose(a_log_prob, b_log_prob)

# def test_benoulli_layer():
#     # Setup layer
#     dim_y = 50
#     dim_x = 100
#
#     mlp = MLP([Logistic()], [dim_y, dim_x], **inits)
#
#     l = BernoulliLayer(mlp, name="layer", **inits)
#     l.initialize()
#
#     y = tensor.fmatrix('y')
#     x_expected = l.sample_expected(y)
#     x, x_log_prob = l.sample(y)
#
#     do = theano.function([y],
#                          [x_expected, x, x_log_prob],
#                          allow_input_downcast=True)
#
#     y = numpy.eye(50, dtype=numpy.float32)
#
#     x_expected, x, x_log_prob = do(y)
#
#     assert x_expected.shape == (50, dim_x)
#     assert x.shape == (50, dim_x)
#     assert x_log_prob.shape == (50,)
