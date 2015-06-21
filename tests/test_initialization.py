
import unittest 
import numpy
import theano

from numpy.testing import assert_equal, assert_allclose, assert_raises

from helmholtz.initialization import *


def test_tanh_initialization():
    def check_tanh(rng, shape):
        weights = TanhInitialization().generate(rng, shape)

        assert weights.shape == shape
        assert weights.dtype == theano.config.floatX
        assert_allclose(weights.mean(), 0, atol=1e-2)

    rng = numpy.random.RandomState(1)
    yield check_tanh, rng, (100, 200)
    yield check_tanh, rng, (200, 100)


def test_rws_initialization():
    def check_tanh(rng, shape):
        weights = RWSInitialization().generate(rng, shape)

        assert weights.shape == shape
        assert weights.dtype == theano.config.floatX
        assert_allclose(weights.mean(), 0, atol=1e-2)

    rng = numpy.random.RandomState(1)
    yield check_tanh, rng, (100, 200)
    yield check_tanh, rng, (200, 100)
