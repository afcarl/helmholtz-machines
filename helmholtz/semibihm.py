
from __future__ import division, print_function

import sys
sys.path.append("../")

import re
import logging

import numpy
import theano

from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Tanh, Logistic
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.select import Selector
from blocks.roles import PARAMETER
 
from initialization import RWSInitialization
from helmholtz import HelmholtzMachine
from helmholtz import merge_gradients, replicate_batch, logsumexp, logplusexp, flatten_values, unflatten_values 
from prob_layers import ProbabilisticTopLayer
logger = logging.getLogger(__name__)
floatX = theano.config.floatX
import numpy as np


def layers_sample(layers, h):
    """ Sample from a stack of conditional probability layers.

    Given a sequence of N layers and an activation for the 
    bottom most activation, sample activations from the 
    posteriors of all N layers.

    Parameters
    ----------
    layers : list of ProbabilisticLayers
    h : Tensor 

    Returns
    -------
    samples : list of Tensors (activations for all layers)
    """
    samples = [None] * len(layers)

    for l,layer in enumerate(layers):
        h, _ = layer.sample(h)
        samples[l] = h

    return samples


def layers_log_prob(layers, samples):
    """ Calculate log_probs for a stack of conditional layers.

    Parameters
    ----------
    layers : list of ProbabilisticLayers
    samples : list of Tensor

    Returns
    -------
    log_probs : list of log_probs
    """
    assert len(layers) == len(samples)-1

    log_probs = [layer.log_prob(x, y) for layer, x, y in zip(layers, samples[1:], samples[:-1])]

    return log_probs


def layers_get_gradients(layers, samples, weights):
    """ Calculate gradients for a stack of conditional layers.

    Parameters
    ----------
    layers : list of ProbabilisticLayers
    samples : list of Tensor
    weights : Yensor

    Returns
    -------
    updates : dict[param] = gradient
    """
    assert len(layers) == len(samples)-1

    gradients = {}
    for layer, x, y in zip(layers, samples[1:], samples[:-1]):
        gradients = merge_gradients(gradients, layer.get_gradients(x, y, weights))

    return gradients
    

def rev(l):
    return l[::-1]

#-----------------------------------------------------------------------------


class SemiBiHM(HelmholtzMachine):
    def __init__(self, bottom_p, bottom_q, top_p, **kwargs):
        super(SemiBiHM, self).__init__(bottom_p, bottom_q, **kwargs)

        assert len(bottom_p) == len(bottom_q)
        assert isinstance(top_p, ProbabilisticTopLayer)

        self.bottom_p = bottom_p
        self.bottom_q = bottom_q
        self.top_p = top_p

        self.px_weight = 1.0

        self.children = bottom_p + bottom_q + [top_p]


    @application(inputs=['features', 'targets', 'mask', 'n_samples'], outputs=['gradients', 'log_px', 'log_pxy', 'log_pygx'])
    def get_gradients(self, features, targets, mask, n_samples):
        bottom_p = self.bottom_p
        bottom_q = self.bottom_q
        n_layers = len(bottom_p)
        top_p = self.top_p

        batch_size = features.shape[0]


        # replicate everything
        x = replicate_batch(features, n_samples)
        y = replicate_batch(targets, n_samples)
        m = replicate_batch(mask, n_samples)
        m = m.reshape((n_samples*batch_size, ))

        mask = mask.reshape((batch_size, ))

        # calculate ingredients for A(x)
        a_samples = layers_sample(bottom_q, x)      # samples upwards from Q
        a_log_probs_q = layers_log_prob(bottom_q, [x]+a_samples)
        a_log_probs_p = rev(layers_log_prob(rev(bottom_p), rev([x]+a_samples)))
        a_log_probs_p += [top_p.log_prob(a_samples[-1])]
        
        a_log_omega = (sum(a_log_probs_p) - sum(a_log_probs_q)) / 2.

        # calculate ingredients for B(x, y)
        #b_samples = layers_sample(bottom_q, x)     # samples upwards from Q
        b_samples = list(a_samples)                 # make copy of samples
        b_samples[-1] = y                           # fix top level to training data
        b_log_probs_q = layers_log_prob(bottom_q, [x]+b_samples)    
        b_log_probs_p = rev(layers_log_prob(rev(bottom_p), rev([x]+b_samples)))
        b_log_probs_p += [top_p.log_prob(b_samples[-1])]

        b_log_omega = (sum(b_log_probs_p) + b_log_probs_q[-1] - sum(b_log_probs_q[:-1])) / 2.

        # mixture proposal for A(x)
        aa_log_probs_q = list(a_log_probs_q)
        aa_log_probs_q[-1] = tensor.switch(tensor.eq(a_samples[-1], y).prod(axis=1),
                                            logplusexp(a_log_probs_q[-1], np.log(1)),
                                            a_log_probs_q[-1])
        ab_log_probs_q = list(b_log_probs_q)
        ab_log_probs_q[-1] = logplusexp(b_log_probs_q[-1], np.log(1))

        aa_log_omega = (sum(a_log_probs_p) - sum(aa_log_probs_q)) / 2.
        ab_log_omega = (sum(b_log_probs_p) - sum(ab_log_probs_q)) / 2.

        # calculate normalized importance weights
        # normalize weights // both a_log_omega and b_log_omrga are (batch_size*n_samples) vectors; we need to reshape and logsumexp

        a_log_omega = a_log_omega.reshape([batch_size, n_samples])
        b_log_omega = b_log_omega.reshape([batch_size, n_samples])
        aa_log_omega = aa_log_omega.reshape([batch_size, n_samples])
        ab_log_omega = ab_log_omega.reshape([batch_size, n_samples])

        log_a  = logsumexp(a_log_omega, axis=1)  # not yet /n_samples!
        log_b  = logsumexp(b_log_omega, axis=1)  # not yet /n_samples!
        log_aa = logplusexp(logsumexp(aa_log_omega, axis=1), logsumexp(ab_log_omega, axis=1)) # not yet /n_samples!

        a_norm_omega = tensor.exp(a_log_omega - tensor.shape_padright(log_a))
        b_norm_omega = tensor.exp(b_log_omega - tensor.shape_padright(log_b))
        aa_norm_omega = tensor.exp(aa_log_omega - tensor.shape_padright(log_aa))
        ab_norm_omega = tensor.exp(ab_log_omega - tensor.shape_padright(log_aa))

        a_norm_omega = a_norm_omega.reshape([batch_size*n_samples])
        b_norm_omega = b_norm_omega.reshape([batch_size*n_samples])
        aa_norm_omega = aa_norm_omega.reshape([batch_size*n_samples])
        ab_norm_omega = ab_norm_omega.reshape([batch_size*n_samples])

        # Choose how to combine the gradients
        #a_norm_omega = tensor.switch(m, -a_norm_omega, 2*a_norm_omega)
        a_norm_omega = tensor.switch(m,                -aa_norm_omega, 2*a_norm_omega*self.px_weight)
        b_norm_omega = tensor.switch(m, -ab_norm_omega + b_norm_omega, 0)

        # calculate gradients
        gradients = OrderedDict()
        gradients = merge_gradients(gradients, layers_get_gradients(bottom_q, [x]+a_samples, a_norm_omega))
        gradients = merge_gradients(gradients, layers_get_gradients(bottom_q, [x]+b_samples, b_norm_omega))
        gradients = merge_gradients(gradients, layers_get_gradients(rev(bottom_p), rev([x]+b_samples), b_norm_omega))
        gradients = merge_gradients(gradients, layers_get_gradients(rev(bottom_p), rev([x]+a_samples), a_norm_omega))
        gradients = merge_gradients(gradients, top_p.get_gradients(a_samples[-1], a_norm_omega))
        gradients = merge_gradients(gradients, top_p.get_gradients(b_samples[-1], b_norm_omega))
    
        # gradients = merge_gradients(gradients, bottom_p[-1].get_gradients(a_samples[-2], a_samples[-1], 1/10))

        # Calculate p(x) and p(y|x)
        log_a -= tensor.log(n_samples)
        log_b -= tensor.log(n_samples)

        log_px   = 2*log_a
        log_pxy  = tensor.switch(mask, log_a + log_b, 0)
        log_pygx = tensor.switch(mask, log_b - log_a, 0)
        #log_pxy  = log_a + log_b
        #log_pygx = log_b - log_a
    
        #import ipdb; ipdb.set_trace()

        return gradients, log_px, log_pxy, log_pygx


    @application(inputs=['features', 'labels', 'mask'], outputs=['log_px', 'log_pxy', 'log_pygx'])
    def log_likelihood(self, features, labels, mask, n_samples):
        """
            features:
            label:
            mask: is 1 for an example with label; 0 when without label
        """
        _, log_px, log_pxy, log_pygx = self.get_gradients(features, labels, mask, n_samples)

        return log_px, log_pxy, log_pygx
        


    def onehot(self, x, numclasses=10):
        if x.shape==():
            x = x[None]
        if numclasses is None:
            numclasses = x.max() + 1
        result = np.zeros(list(x.shape) + [numclasses], dtype="int32")
        z = np.zeros(x.shape)
        for c in range(numclasses):
            z *= 0
            z[np.where(x==c)] = 1
            result[...,c] += z
        return result
