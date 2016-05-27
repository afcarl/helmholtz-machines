
from __future__ import division, print_function

import logging

import numpy
import theano

from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.roles import has_roles, WEIGHT
from blocks.select import Selector

from . import HelmholtzMachine
from . import merge_gradients, flatten_values, unflatten_values, replicate_batch, logsumexp, logplusexp

from prob_layers import BernoulliTopLayer, BernoulliLayer

from initialization import RWSInitialization
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.bricks import Random, Initializable, MLP, Tanh, Logistic

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

n_steps = 14

#-----------------------------------------------------------------------------

def rev(samples):
    return [h[::-1] for h in samples]

def sample_given_x(x, layers0, layers):
    n_samples = x.shape[0]

    h0 = [x[:, i, :]                                 for i in xrange(n_steps)]
    h1 = [tensor.zeros([n_samples, layers[1].dim_X]) for i in xrange(n_steps+1)]
    h2 = [tensor.zeros([n_samples, layers[2].dim_X]) for i in xrange(n_steps+2)]

    log_prob0 = [None for _ in xrange(n_steps)]
    log_prob1 = [None for _ in xrange(n_steps+1)]
    log_prob2 = [None for _ in xrange(n_steps+2)]

    # step 0
    log_prob0[0] = layers0[0].log_prob(h0[0])
    #log_prob0[0] = tensor.zeros([n_samples])
    h1[1], log_prob1[1] = layers0[1].sample(h0[0])
    h2[2], log_prob2[2] = layers0[2].sample(h1[1])

    # step 1...N
    for step in xrange(1, n_steps):
        c0 = tensor.concatenate([            h0[step-1], h1[step+0]], axis=1)
        log_prob0[step+0] = layers[0].log_prob(h0[step], c0)
        #log_prob0[step+0] = tensor.zeros([n_samples])

        c1 = tensor.concatenate([h0[step+0], h1[step+0], h2[step+1]], axis=1)
        h1[step+1], log_prob1[step+1] = layers[1].sample(c1)

        c2 = tensor.concatenate([h1[step+1], h2[step+1]            ], axis=1)
        h2[step+2], log_prob2[step+2] = layers[2].sample(c2)

    return [h0, h1, h2], [log_prob0, log_prob1, log_prob2]

def sample_sequence(n_samples, layers0, layers):

    h0 = [None                                       for i in xrange(n_steps)]
    h1 = [tensor.zeros([n_samples, layers[1].dim_X]) for i in xrange(n_steps+1)]
    h2 = [tensor.zeros([n_samples, layers[2].dim_X]) for i in xrange(n_steps+2)]

    log_prob0 = [None for _ in xrange(n_steps)]
    log_prob1 = [None for _ in xrange(n_steps+1)]
    log_prob2 = [None for _ in xrange(n_steps+2)]

    # step 0
    h0[0], log_prob0[0] = layers0[0].sample(n_samples)
    h1[1], log_prob1[1] = layers0[1].sample(h0[0])
    h2[2], log_prob2[2] = layers0[2].sample(h1[1])

    # step 1...N
    for step in xrange(1, n_steps):
        c0 = tensor.concatenate([            h0[step-1], h1[step+0]], axis=1)
        h0[step+0], log_prob0[step+0] = layers[0].sample(c0)

        c1 = tensor.concatenate([h0[step+0], h1[step+0], h2[step+1]], axis=1)
        h1[step+1], log_prob1[step+1] = layers[1].sample(c1)

        c2 = tensor.concatenate([h1[step+1], h2[step+1]            ], axis=1)
        h2[step+2], log_prob2[step+2] = layers[2].sample(c2)

    return [h0, h1, h2], [log_prob0, log_prob1, log_prob2]


def log_prob_sequence(samples, layers0, layers):

    h0, h1, h2 = samples

    log_prob0 = [None for _ in xrange(n_steps)]
    log_prob1 = [None for _ in xrange(n_steps+1)]
    log_prob2 = [None for _ in xrange(n_steps+2)]

    # step 0
    log_prob0[0] = layers0[0].log_prob(h0[0])
    log_prob1[1] = layers0[1].log_prob(h1[1], h0[0])
    log_prob2[2] = layers0[2].log_prob(h2[2], h1[1])

    # step 1...N
    for step in xrange(1, n_steps):
        c0 = tensor.concatenate([            h0[step-1], h1[step+0]], axis=1)
        log_prob0[step+0] = layers[0].log_prob(h0[step+0], c0)

        c1 = tensor.concatenate([h0[step+0], h1[step+0], h2[step+1]], axis=1)
        log_prob1[step+1] = layers[1].log_prob(h1[step+1], c1)

        c2 = tensor.concatenate([h1[step+1], h2[step+1]            ], axis=1)
        log_prob2[step+2] = layers[2].log_prob(h2[step+2], c2)

    return log_prob0, log_prob1, log_prob2


#-----------------------------------------------------------------------------

class BiSeq(HelmholtzMachine):
    def __init__(self, layer_sizes, n_steps, **kwargs):
        super(BiSeq, self).__init__([], [], **kwargs)

        assert len(layer_sizes) == 3
        assert layer_sizes[0] * n_steps == 28*28

        self.n_steps = n_steps
        self.layer_sizes = layer_sizes

        inits = {
            'weights_init': RWSInitialization(factor=1.),
            #        'weights_init': IsotropicGaussian(0.1),
            'biases_init': Constant(-1.0),
        }

        dim_c1 = layer_sizes[0]
        dim_c2 = layer_sizes[1]
        self.p_layers0 = [
                BernoulliTopLayer(layer_sizes[0], name="p00", **inits),
                BernoulliLayer(MLP([Logistic()], [dim_c1, layer_sizes[1]], **inits), name="p01"),
                BernoulliLayer(MLP([Logistic()], [dim_c2, layer_sizes[2]], **inits), name="p02"),
            ]
        self.q_layers0 = [
                BernoulliTopLayer(layer_sizes[0], name="q00", **inits),
                BernoulliLayer(MLP([Logistic()], [dim_c1, layer_sizes[1]], **inits), name="q01"),
                BernoulliLayer(MLP([Logistic()], [dim_c2, layer_sizes[2]], **inits), name="q02"),
            ]


        dim_c0 = layer_sizes[0] + layer_sizes[1]
        dim_c1 = layer_sizes[0] + layer_sizes[1] + layer_sizes[2]
        dim_c2 = layer_sizes[1] + layer_sizes[2]
        self.p_layers = [
                BernoulliLayer(MLP([Logistic()], [dim_c0, layer_sizes[0]], **inits), name="p0"),
                BernoulliLayer(MLP([Logistic()], [dim_c1, layer_sizes[1]], **inits), name="p1"),
                BernoulliLayer(MLP([Logistic()], [dim_c2, layer_sizes[2]], **inits), name="p2"),
            ]
        self.q_layers = [
                BernoulliLayer(MLP([Logistic()], [dim_c0, layer_sizes[0]], **inits), name="q0"),
                BernoulliLayer(MLP([Logistic()], [dim_c1, layer_sizes[1]], **inits), name="q1"),
                BernoulliLayer(MLP([Logistic()], [dim_c2, layer_sizes[2]], **inits), name="q2"),
            ]


        self.children = self.p_layers + self.p_layers0 + self.q_layers + self.q_layers0


    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx'])
    def log_likelihood(self, features, n_samples):
        log_ps, _ = self.get_gradients(features, n_samples)

        return log_ps


    @application(inputs=['features', 'n_samples'], outputs=['log_px', 'log_psx', 'gradients'])
    def get_gradients(self, features, n_samples):
        """Perform inference and calculate gradients.

        Returns
        -------
        log_ps : T.fvector
        gradients : OrderedDict
        """
        p_layers  = self.p_layers
        p_layers0 = self.p_layers0
        q_layers  = self.q_layers
        q_layers0 = self.q_layers0
        n_layers  = len(p_layers)

        n_steps = self.n_steps
        layer_sizes = self.layer_sizes

        batch_size = features.shape[0]
        total_size = batch_size * n_samples

        x = replicate_batch(features, n_samples)
        x = x.reshape([batch_size*n_samples, n_steps, layer_sizes[0]])

        samples_p, log_pp = sample_given_x(x            , p_layers0, p_layers)
        samples_q, log_qq = sample_given_x(x[:, ::-1, :], q_layers0, q_layers)
        samples_q = rev(samples_q)
        log_qq = rev(log_qq)

        log_qp = log_prob_sequence(samples_q     , p_layers0, p_layers)
        log_pq = log_prob_sequence(rev(samples_p), q_layers0, q_layers)
        log_pq = rev(log_pq)

        # 
        samples_p = [h[i:len(h)-i] for i, h in enumerate(samples_p)]
        samples_q = [h[i:len(h)-i] for i, h in enumerate(samples_q)]

        log_pp = [lp[i:len(lp)-i] for i, lp in enumerate(log_pp)]
        log_pq = [lp[i:len(lp)-i] for i, lp in enumerate(log_pq)]
        log_qp = [lp[i:len(lp)-i] for i, lp in enumerate(log_qp)]
        log_qq = [lp[i:len(lp)-i] for i, lp in enumerate(log_qq)]

        log_p_prop = sum([sum(lp) for lp in log_pp[1:]]).reshape([batch_size, n_samples])
        log_q_prop = sum([sum(lp) for lp in log_qq[1:]]).reshape([batch_size, n_samples])

        log_pp = sum([sum(lp) for lp in log_pp]).reshape([batch_size, n_samples])
        log_pq = sum([sum(lp) for lp in log_pq]).reshape([batch_size, n_samples])
        log_qp = sum([sum(lp) for lp in log_qp]).reshape([batch_size, n_samples])
        log_qq = sum([sum(lp) for lp in log_qq]).reshape([batch_size, n_samples])

        log_ps_p = (log_pp + log_pq) / 2                 # shape batch x n_samples 
        log_ps_q = (log_qp + log_qq) / 2                 # shape batch x n_samples

        log_psmq_p = log_ps_p - log_p_prop               # shape batch x n_samples
        log_psmq_q = log_ps_q - log_q_prop               # shape batch x n_samples

        log_ps   = tensor.concatenate([log_ps_p  , log_ps_q]  , axis=1) # shape batch x 2 n_samples
        log_psmq = tensor.concatenate([log_psmq_p, log_psmq_q], axis=1) # shape batch x 2 n_samples

        log_px = logsumexp(log_psmq, axis=1) - tensor.log(2*n_samples)  # shape batch 
        w_norm = logsumexp(log_psmq, axis=1)                            # shape batch 
        log_w = log_psmq - tensor.shape_padright(w_norm)                # shape batch x 2 n_samples
        w = tensor.exp(log_w)                                           # shape batch x 2 n_samples

        log_ps = log_ps.reshape([batch_size*2*n_samples])               # shape N
        w = w.reshape([batch_size*2*n_samples])                         # shape N

        # samples
        h0_p, h1_p, h2_p = samples_p
        h0_q, h1_q, h2_q = samples_q
        h0, h1, h2 = h0_p + h0_q, h1_p + h1_q, h2_p + h2_q

        # Calculate gradients
        gradients = OrderedDict()

        # cost
        grad_cost = w * log_ps
        grad_cost = -grad_cost.sum()

        params = Selector(self).get_parameters()
        for pname, param in params.iteritems():
            gradients[param] = tensor.grad(grad_cost, param, consider_constant=h0+h1+h2+[w])

        return log_px, gradients

    #==================================================================================

    def estimate_log_z(self, n_samples):
        p_layers  = self.p_layers
        p_layers0 = self.p_layers0
        q_layers  = self.q_layers
        q_layers0 = self.q_layers0
        n_layers  = len(p_layers)

        # P proposals
        samples, log_prob_p = sample_sequence(n_samples, p_layers0, p_layers)
        log_prob_q = log_prob_sequence(rev(samples), q_layers0, q_layers)

        log_prob_p0, log_prob_p1, log_prob_p2 = log_prob_p
        log_prob_q0, log_prob_q1, log_prob_q2 = log_prob_q

        log_prob_p = sum(log_prob_p0) + sum(log_prob_p1[1:-1]) + sum(log_prob_p2[2:-2])
        log_prob_q = sum(log_prob_q0) + sum(log_prob_q1[1:-1]) + sum(log_prob_q2[2:-2])

        log_prob1  = (log_prob_q - log_prob_p) / 2

        # Q proposals
        samples, log_prob_q = sample_sequence(n_samples, q_layers0, q_layers)
        log_prob_p = log_prob_sequence(rev(samples), p_layers0, p_layers)

        log_prob_q0, log_prob_q1, log_prob_q2 = log_prob_q
        log_prob_p0, log_prob_p1, log_prob_p2 = log_prob_p

        log_prob_p = sum(log_prob_p0) + sum(log_prob_p1[1:-1]) + sum(log_prob_p2[2:-2])
        log_prob_q = sum(log_prob_q0) + sum(log_prob_q1[1:-1]) + sum(log_prob_q2[2:-2])

        log_prob2  = (log_prob_p - log_prob_q) / 2

        return log_prob1, log_prob2


    @application(inputs=['n_samples'], outputs=['samples'])
    def sample_p(self, n_samples):
        """
        """
        p_layers  = self.p_layers
        p_layers0 = self.p_layers0

        samples, _ = sample_sequence(n_samples, p_layers0, p_layers)

        return samples

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample_q(self, n_samples):
        """
        """
        q_layers  = self.q_layers
        q_layers0 = self.q_layers0

        samples, _ = sample_sequence(n_samples, q_layers0, q_layers)
        samples = rev(samples)

        return samples
   
   
    # #@application(inputs=['features'],
    # #             outputs=['samples', 'log_q', 'log_p'])
    # def sample_q(self, features):
    #     p_layers = self.p_layers
    #     q_layers = self.q_layers
    #     n_layers = len(p_layers)
    #
    #     batch_size = features.shape[0]
    #
    #     samples = [None] * n_layers
    #     log_p = [None] * n_layers
    #     log_q = [None] * n_layers
    #
    #     # Generate samples (feed-forward)
    #     samples[0] = features
    #     log_q[0] = tensor.zeros([batch_size])
    #     for l in xrange(n_layers - 1):
    #         samples[l + 1], log_q[l + 1] = q_layers[l].sample(samples[l])
    #
    #     # get log-probs from generative model
    #     log_p[n_layers - 1] = p_layers[n_layers - 1].log_prob(samples[n_layers - 1])
    #     for l in reversed(range(1, n_layers)):
    #         log_p[l - 1] = p_layers[l - 1].log_prob(samples[l - 1], samples[l])
    #
    #     return samples, log_p, log_q
    #
    #
    # @application(inputs=['n_samples', 'oversample', 'n_inner'],
    #              outputs=['samples', 'log_w'])
    # def sample(self, n_samples, oversample=100, n_inner=10):
    #     p_layers = self.p_layers
    #     q_layers = self.q_layers
    #     n_layers = len(p_layers)
    #
    #     n_primary = n_samples * oversample
    #
    #     samples, log_p, log_q = self.sample_p(n_primary)
    #
    #     # Sum all layers
    #     log_p_all = sum(log_p)   # This is the python sum over a list
    #     log_q_all = sum(log_q)   # This is the python sum over a list
    #
    #     _, log_qx = self.log_likelihood(samples[0], n_inner)
    #
    #     log_w = (log_qx + log_q_all - log_p_all) / 2
    #     w_norm = logsumexp(log_w, axis=0)
    #     log_w = log_w - w_norm
    #     w = tensor.exp(log_w)
    #
    #     pvals = w.dimshuffle('x', 0).repeat(n_samples, axis=0)
    #     idx = self.theano_rng.multinomial(pvals=pvals).argmax(axis=1)
    #
    #     subsamples = [s[idx, :] for s in samples]
    #
    #     return subsamples, log_w
