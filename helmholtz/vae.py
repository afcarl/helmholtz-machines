
from __future__ import division, print_function 

import sys
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

from helmholtz import HelmholtzMachine
from prob_layers import replicate_batch, logsumexp
from prob_layers import GaussianLayer, BernoulliLayer
from initialization import RWSInitialization

logger = logging.getLogger(__name__)
floatX = theano.config.floatX


#-----------------------------------------------------------------------------


class VAE(HelmholtzMachine):
    def __init__(self, x_dim, hidden_layers, z_dim, **kwargs):
        super(VAE, self).__init__([], [], **kwargs)

        inits = {
            'weights_init': RWSInitialization(factor=1.),
#            'weights_init': IsotropicGaussian(0.1),
            'biases_init': Constant(-1.0),
        }

        hidden_act = [Tanh()]*len(hidden_layers)

        q_mlp = MLP(hidden_act             , [x_dim]+hidden_layers, **inits)
        p_mlp = MLP(hidden_act+[Logistic()], [z_dim]+hidden_layers+[x_dim], **inits)

        self.q = GaussianLayer(z_dim, q_mlp, **inits)
        self.p = BernoulliLayer(p_mlp, **inits)

        self.children = [self.p, self.q]

        self.prior_log_sigma = numpy.zeros(z_dim)
        self.prior_mu = numpy.zeros(z_dim)


    #@application(inputs=['n_samples'], 
    #             outputs=['samples', 'log_q', 'log_p'])
    def sample(self, n_samples, oversample=100, n_inner=10):
        return

    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_px'])
    def log_likelihood(self, features, n_samples):
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)

        z, log_q = self.q.sample(x)
        log_p = self.p.log_prob(x, z)

        log_q = log_q.reshape([batch_size, n_samples])
        log_p = log_p.reshape([batch_size, n_samples])
        log_px = logsumexp(log_p-log_q, axis=1) - tensor.log(n_samples)
        #log_px = log_p - log_q

        return log_px, log_px


    @application(inputs=['features', 'n_samples'], outputs=['log_p_bound'])
    def log_likelihood_bound(self, features, n_samples):
        """ 
        Computye the LL bound
        """
        batch_size = features.shape[0]

        z_mu, z_log_sigma = self.q.sample_expected(features)

        # Recosntruction...
        features_r    = replicate_batch(features, n_samples)
        z_mu_r        = replicate_batch(z_mu, n_samples)
        z_log_sigma_r = replicate_batch(z_log_sigma, n_samples)

        epsilon = self.theano_rng.normal(size=z_mu_r.shape, dtype=z_mu_r.dtype)
        z_r = z_mu_r + epsilon * tensor.exp(z_log_sigma_r)

        recons_term = self.p.log_prob(features_r, z_r)
        recons_term = recons_term.reshape([batch_size, n_samples])
        recons_term = tensor.sum(recons_term, axis=1) / n_samples

        # KL divergence
        per_dim_kl = (
                self.prior_log_sigma - z_log_sigma 
                + 0.5 * (
                    tensor.exp(2*z_log_sigma) + (z_mu - self.prior_mu)**2
                ) / tensor.exp(2*self.prior_log_sigma)
                - 0.5)
    
        kl_term = per_dim_kl.sum(axis=1)
        kl_term.name = 'kl_term'

        log_p_bound = recons_term - kl_term

        return log_p_bound

