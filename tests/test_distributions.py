
from __future__ import division, print_function

import unittest

import numpy

import theano
import theano.tensor as tensor

from theano.sandbox.rng_mrg import MRG_RandomStreams

import helmholtz.distributions as dist


def test_benoulli():
    theano_rng = MRG_RandomStreams(seed=2341)

    n_samples = tensor.iscalar("n_samples")
    prob = tensor.vector('prob')
    target_prob = tensor.vector('target_prob')

    shape = (n_samples, prob.shape[0])
    bprob = tensor.ones(shape) * prob

    samples = dist.bernoulli(bprob, rng=theano_rng)

    mean = tensor.mean(samples, axis=0)
    cost = tensor.sum((mean-target_prob)**2)

    grads = theano.grad(cost, prob)

    print("-"*78)
    print(theano.printing.debugprint(samples))
    print("-"*78)

    do_sample = theano.function(
                inputs=[prob, target_prob, n_samples],
                outputs=[samples, grads],
                allow_input_downcast=True, name="do_sample")

    #-------------------------------------------------------------------------
    n_samples = 10000
    prob = numpy.linspace(0, 1, 11)
    target_prob = prob

    samples, grads = do_sample(prob, target_prob, n_samples)
    print("== samples =========")
    print(samples)
    print("== mean ============")
    print(numpy.mean(samples, axis=0))
    print("== grads ===========")
    print(grads)

    assert numpy.allclose(numpy.mean(samples, axis=0), prob, atol=0.1, rtol=0.1)
