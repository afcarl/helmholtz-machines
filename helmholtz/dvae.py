
from __future__ import division, print_function 

import sys
import logging

import numpy
import theano

from theano import tensor

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Logistic
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity

from helmholtz import HelmholtzMachine
from prob_layers import replicate_batch, logsumexp
from prob_layers import BernoulliLayer, BernoulliTopLayer
from initialization import RWSInitialization

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

#-----------------------------------------------------------------------------


class DVAE(HelmholtzMachine, Random):
    """ Discrete Variational Autoencoder with
            p(z) = Bernoulli(z | 0.5)  and
            p(x|z) = Bernoulli( ... )
    """
    def __init__(self, x_dim, hidden_layers, hidden_act, z_dim, **kwargs):
        super(DVAE, self).__init__([], [], **kwargs)

        inits = {
            #'weights_init': IsotropicGaussian(std=0.1),
            'weights_init': RWSInitialization(factor=1.),
            'biases_init': Constant(0.0),
        }

        hidden_act = [hidden_act]*len(hidden_layers)

        q_mlp = MLP(hidden_act+[Logistic()], [x_dim]+hidden_layers+[z_dim], **inits)
        p_mlp = MLP(hidden_act+[Logistic()], [z_dim]+hidden_layers+[x_dim], **inits)

        self.q = BernoulliLayer(q_mlp, name="q")
        self.p = BernoulliLayer(p_mlp, name="p")
        self.p_top = BernoulliTopLayer(z_dim, **inits)

        self.children = [self.p_top, self.p, self.q]


    @application(inputs=['n_samples'], 
                 outputs=['samples'])
    def sample(self, n_samples):
        # Sample from mean-zeros std.-one Gaussian
        eps = self.theano_rng.uniform(
                    size=(n_samples, self.dim_z),
                    low=0., high=1.)

        raise NotImplemented
        #z = self.prior_mu + tensor.exp(seld.prior_log_sigma) * eps
        # 
        #x, _ = self.p.sample(z)
        #return [x, z]


    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_px'])
    def log_likelihood(self, features, n_samples):
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)

        z, log_q = self.q.sample(x)
        log_p = self.p.log_prob(x, z)     # p(x|z)
        log_p += self.p_top.log_prob(z)   # p(z) prior

        log_q = log_q.reshape([batch_size, n_samples])
        log_p = log_p.reshape([batch_size, n_samples])
        log_px = logsumexp(log_p-log_q, axis=1) - tensor.log(n_samples)

        return log_px, log_px


    @application(inputs=['features', 'n_samples'], outputs=['log_p_bound'])
    def log_likelihood_bound(self, features, n_samples=1):
        """Compute the LL bound. """
        batch_size = features.shape[0]

        z_prob = self.q.sample_expected(features)

        # Recosntruction...
        features_r = replicate_batch(features, n_samples)
        z_prob_r   = replicate_batch(z_prob, n_samples)

        rho = self.theano_rng.uniform(
                    size=z_prob_r.shape, 
                    low=0, high=1.,
                    dtype=z_prob.dtype) 

        z_r = tensor.switch(rho >= 1-z_prob_r, (rho-1)/z_prob_r + 1, 0.)

        recons_term = self.p.log_prob(features_r, z_r)
        recons_term = recons_term.reshape([batch_size, n_samples])
        recons_term = tensor.sum(recons_term, axis=1) / n_samples

        # KL divergence
        prior_prob = self.p_top.sample_expected()

        per_dim_kl = (
                prior_prob * tensor.log(prior_prob/z_prob) +
                (1-prior_prob) * tensor.log((1-prior_prob)/(1-z_prob)))
        kl_term = per_dim_kl.sum(axis=1)
        kl_term.name = 'kl_term'

        log_p_bound = recons_term - kl_term

        return log_p_bound
