#!/usr/bin/env python 

from __future__ import division, print_function

import numpy as np

import theano 
import theano.tensor as T

from blocks.bricks import MLP, Tanh, Logistic, Rectifier
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse 

from helmholtz.prob_layers import MultinomialTopLayer, MultinomialLayer



inits = {
    'biases_init': Constant(0.0),
    'weights_init': IsotropicGaussian()
}


def test_multinomial_layer():
    n_samples = T.iscalar('n_samples')
    y = T.fmatrix('y')

    l2 = MultinomialLayer(MLP([Logistic()], [2, 3], **inits))
    l2.initialize()

    z_expected = l2.sample_expected(y)
    z_samples, _ = l2.sample(y)

    do = theano.function([y], [z_expected, z_samples])
  
    y = np.array([[0., 1.]]*10 + [[1., 0.]]*10, dtype=np.float32)

    z_expected, z_samples = do(y)
    #print(z_expected)
    #print(z_samples)


def test_multinomial_top_layer():
    n_samples = T.iscalar('n_samples')
    y = T.fmatrix('y')

    l1 = MultinomialTopLayer(3, **inits)
    l1.initialize()

    x_expected = l1.sample_expected(n_samples)
    x_samples, _ = l1.sample(n_samples)

    do = theano.function([n_samples], [x_expected, x_samples])

    x_expected, x_samples = do(100)
    #print(x_expected)
    #print(x_samples)
