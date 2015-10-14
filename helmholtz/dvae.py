
from __future__ import division, print_function 

import sys
import logging

import numpy
import theano

from theano import tensor

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Logistic
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity

from batch_normalization import BatchNormalization, BatchNormalizedMLP
from helmholtz import HelmholtzMachine
from initialization import RWSInitialization
from prob_layers import replicate_batch, logsumexp
from prob_layers import BernoulliLayer, BernoulliTopLayer

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
            'weights_init': IsotropicGaussian(std=0.1),
            #'weights_init': RWSInitialization(factor=1.),
            'biases_init': Constant(0.0),
        }

        hidden_act = [hidden_act]*len(hidden_layers)

        q_mlp = BatchNormalizedMLP(hidden_act+[Logistic()], [x_dim]+hidden_layers+[z_dim], **inits)
        #q_mlp = MLP(hidden_act+[Logistic()], [x_dim]+hidden_layers+[z_dim], **inits)
        p_mlp = BatchNormalizedMLP(hidden_act+[Logistic()], [z_dim]+hidden_layers+[x_dim], **inits)
        #p_mlp = MLP(hidden_act+[Logistic()], [z_dim]+hidden_layers+[x_dim], **inits)

        self.q = BernoulliLayer(q_mlp, name="q")
        self.p = BernoulliLayer(p_mlp, name="p")
        self.p_top = BernoulliTopLayer(z_dim, biases_init=Constant(0.0))

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

        prior_z_prob = self.p_top.sample_expected()
        prior_z_prob = prior_z_prob.clip(1e-10, 1-1e-10)
        prior_z_prob = tensor.shape_padleft(prior_z_prob)

        z_prob = self.q.sample_expected(features)
        z_prob = z_prob.clip(1e-10, 1-1e-10)

        z_prob   = replicate_batch(z_prob, n_samples)
        features = replicate_batch(features, n_samples)

        rho = self.theano_rng.uniform(
                    size=z_prob.shape, 
                    low=0, high=1.,
                    dtype=z_prob.dtype) 

        a = (rho-1)/z_prob + 1
        xi = tensor.switch(a > 0., a, 0)

        # q(xi|x) approximate posterior
        log_q = tensor.switch(a > 0., z_prob, 1-z_prob)
        log_q = tensor.log(log_q).sum(axis=1)

        # p(xi) prior
        log_p = tensor.switch(a > 0., prior_z_prob, 1-prior_z_prob)
        log_p = tensor.log(log_p).sum(axis=1)

        # + p(x|xi)
        log_p += self.p.log_prob(features, xi)
       
        log_pq = log_p-log_q
        log_pq = log_pq.reshape([batch_size, n_samples])
        log_px = logsumexp(log_pq, axis=1) - tensor.log(n_samples)

        return log_px, log_px


    @application(inputs=['features', 'n_samples'], outputs=['log_p_bound'])
    def log_likelihood_bound(self, features, n_samples=1):
        """Compute the LL bound. """
        batch_size = features.shape[0]

        z_prob = self.q.sample_expected(features)

        # Reconstruction...
        features_r = replicate_batch(features, n_samples)
        z_prob_r   = replicate_batch(z_prob, n_samples)

        rho = self.theano_rng.uniform(
                    size=z_prob_r.shape, 
                    low=0, high=1.,
                    dtype=z_prob.dtype) 

        xi = (rho-1)/z_prob_r + 1
        z_r = tensor.switch(xi > 0., xi, 0)

        recons_term = self.p.log_prob(features_r, z_r)
        recons_term = recons_term.reshape([batch_size, n_samples])
        recons_term = tensor.sum(recons_term, axis=1) / n_samples
        recons_term.name = 'recons_term'

        # KL divergence
        prior_prob = self.p_top.sample_expected()

        per_dim_kl = (
                   prior_prob  * (tensor.log(  prior_prob)-tensor.log(  z_prob)) +
                (1-prior_prob) * (tensor.log(1-prior_prob)-tensor.log(1-z_prob))
        )
        kl_term = per_dim_kl.sum(axis=1)
        kl_term.name = 'kl_term'

        self.recons_term = recons_term
        self.kl_term = kl_term

        log_p_bound = recons_term - kl_term

        return log_p_bound
