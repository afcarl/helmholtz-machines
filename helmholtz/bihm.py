
from __future__ import division, print_function 

import sys

import re
import logging

import numpy
import theano

from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.select import Selector

from helmholtz import HelmholtzMachine
from helmholtz import merge_gradients, flatten_values, unflatten_values
from prob_layers import replicate_batch, logsumexp

logger = logging.getLogger(__name__)
floatX = theano.config.floatX


#-----------------------------------------------------------------------------


class BiHM(HelmholtzMachine):
    def __init__(self, p_layers, q_layers, zreg=0.0, transpose_init=False, **kwargs):
        super(BiHM, self).__init__(p_layers, q_layers, **kwargs)

        self.transpose_init = transpose_init
        self.zreg = zreg

        self.children = p_layers + q_layers

    def log_prob_p(self, samples):
        """ Calculate p(h_l | h_{l+1}) for all layers.  """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_p = [None] * n_layers
        for l in xrange(n_layers-1):
            log_p[l] = self.p_layers[l].log_prob(samples[l], samples[l+1])
        log_p[n_layers-1] = self.p_layers[n_layers-1].log_prob(samples[n_layers-1])

        return log_p

    def log_prob_q(self, samples):
        """ Calculate q(h_{l+1} | h_l_ for all layers *but the first one*.  """
        n_layers = len(self.p_layers)
        n_samples = samples[0].shape[0]

        log_q = [None] * n_layers
        log_q[0] = tensor.zeros([n_samples])
        for l in xrange(n_layers-1):
            log_q[l+1] = self.q_layers[l].log_prob(samples[l+1], samples[l])

        return log_q

    #@application(inputs=['n_samples'], outputs=['samples', 
    def sample_p(self, n_samples):
        """
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)
        
        samples = [None] * n_layers
        log_p = [None] * n_layers

        samples[n_layers-1], log_p[n_layers-1] = p_layers[n_layers-1].sample(n_samples)
        for l in reversed(xrange(1, n_layers)):
            samples[l-1], log_p[l-1] = p_layers[l-1].sample(samples[l])

        # Get log_q
        log_q = self.log_prob_q(samples)
    
        return samples, log_p, log_q

    #@application(inputs=['features'], 
    #             outputs=['samples', 'log_q', 'log_p'])
    def sample_q(self, features):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        batch_size = features.shape[0]
        
        samples = [None] * n_layers
        log_p = [None] * n_layers
        log_q = [None] * n_layers

        # Generate samples (feed-forward)
        samples[0] = features
        log_q[0] = tensor.zeros([batch_size])
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])

        # get log-probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in reversed(range(1, n_layers)):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])
            
        return samples, log_p, log_q

    #@application(inputs=['n_samples'], 
    #             outputs=['samples', 'log_q', 'log_p'])
    def sample(self, n_samples, oversample=100, n_inner=10):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        n_primary = n_samples*oversample

        samples, log_p, log_q = self.sample_p(n_primary)

        # Sum all layers
        log_p_all = sum(log_p)   # This is the python sum over a list
        log_q_all = sum(log_q)   # This is the python sum over a list

        _, log_qx = self.log_likelihood(samples[0], n_inner)

        log_w = (log_qx + log_q_all - log_p_all) / 2
        w_norm = logsumexp(log_w, axis=0)
        log_w = log_w-w_norm
        w = tensor.exp(log_w)

        #pvals = w.repeat(n_samples, axis=0)
        pvals = w.dimshuffle('x', 0).repeat(n_samples, axis=0)
        idx = self.theano_rng.multinomial(pvals=pvals).argmax(axis=1)

        subsamples = [s[idx,:] for s in samples]
    
        return subsamples, log_w

    def importance_weights(self, samples, log_p, log_q):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        # Sum all layers
        log_p_all = sum(log_p)   # This is the python sum over a list
        log_q_all = sum(log_q)   # This is the python sum over a list
    
        # Calculate sampling weights
        log_pq = (log_p_all-log_q_all)/2
        w_norm = logsumexp(log_pq, axis=1)
        log_w = log_pq-tensor.shape_padright(w_norm)
        w = tensor.exp(log_w)
        
        return w 
        

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx'])
    def log_likelihood(self, features, n_samples):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)
        samples, log_p, log_q = self.sample_q(x)

        # Reshape and sum
        samples = unflatten_values(samples, batch_size, n_samples)
        log_p = unflatten_values(log_p, batch_size, n_samples)
        log_q = unflatten_values(log_q, batch_size, n_samples)

        log_p_all = sum(log_p)
        log_q_all = sum(log_q)

        # Approximate log(p(x))
        log_px  = logsumexp(log_p_all-log_q_all, axis=-1) - tensor.log(n_samples)
        log_psx = (logsumexp((log_p_all-log_q_all)/2, axis=-1) - tensor.log(n_samples)) * 2.

        return log_px, log_psx

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx', 'gradients'])
    def get_gradients(self, features, n_samples):
        """Perform inference and calculate gradients.

        Returns
        -------
            log_px : T.fvector
            log_psx : T.fvector
            gradients : OrderedDict
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)

        # Get Q-samples
        samples, log_p, log_q = self.sample_q(x)

        # Reshape and sum
        samples = unflatten_values(samples, batch_size, n_samples)
        log_p = unflatten_values(log_p, batch_size, n_samples)
        log_q = unflatten_values(log_q, batch_size, n_samples)

        log_p_all = sum(log_p)
        log_q_all = sum(log_q)

        # Approximate log(p(x))
        log_px  = logsumexp(log_p_all-log_q_all, axis=-1) - tensor.log(n_samples)
        log_psx = (logsumexp((log_p_all-log_q_all)/2, axis=-1) - tensor.log(n_samples)) * 2.

        # Approximate log p(x) and calculate IS weights
        w = self.importance_weights(samples, log_p, log_q)
        w = w.reshape( (batch_size*n_samples, ) )
        samples = flatten_values(samples, batch_size*n_samples)

        gradients = OrderedDict()
        for l in xrange(n_layers-1):
            gradients = merge_gradients(gradients, p_layers[l].get_gradients(samples[l], samples[l+1], weights=w))
            gradients = merge_gradients(gradients, q_layers[l].get_gradients(samples[l+1], samples[l], weights=w))
        gradients = merge_gradients(gradients, p_layers[-1].get_gradients(samples[-1], weights=w))


        if self.zreg > 0.0:
            #zreg = batch_size * numpy.float32(self.zreg)
            #zreg = numpy.float32(self.zreg)
            #zreg = zreg.astype(floatX)
            zreg = numpy.cast[floatX](self.zreg)

            # And go down again...
            samples, log_p, log_q = self.sample_p(batch_size)

            # Sum all layers
            log_p_all = sum(log_p)   # This is the python sum over a list
            log_q_all = sum(log_q)   # This is the python sum over a list

            _, log_q0 = self.log_likelihood(samples[0], n_samples)

            log_w = (log_q0 + log_q_all - log_p_all) / 2
            w_norm = logsumexp(log_w, axis=0)
            log_w = log_w-w_norm
            w = tensor.exp(log_w)

            for l in xrange(n_layers-1):
                gradients = merge_gradients(gradients, 
                            p_layers[l].get_gradients(samples[l], samples[l+1], weights=w), 
                            scale=zreg)
                gradients = merge_gradients(gradients,
                            q_layers[l].get_gradients(samples[l+1], samples[l], weights=w),
                            scale=zreg)
 

        return log_px, log_psx, gradients

        """
        if True:
            cost = 0
            for l in xrange(n_layers-1):
                cost = cost - (w * p_layers[l].log_prob(samples[l], samples[l+1])).mean()
                cost = cost - (w * q_layers[l].log_prob(samples[l+1], samples[l])).mean()
            cost = cost - (w * p_layers[-1].log_prob(samples[-1])).mean()

            gradients = OrderedDict()
            for l in xrange(n_layers-1):
                for pname, param in Selector(p_layers[l]).get_params().iteritems():
                    gradients[param] = tensor.grad(cost, param)
                for pname, param in Selector(q_layers[l]).get_params().iteritems():
                    gradients[param] = tensor.grad(cost, param)
            for pname, param in Selector(p_layers[-1]).get_params().iteritems():
                gradients[param] = tensor.grad(cost, param)
        else:
            pass
        return log_psx, log_px, gradients
        """

 

        """
       """
