
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

#-----------------------------------------------------------------------------

def rev(samples):
    return [h[::-1] for h in samples]

def sample_given_x(layers0, layers):
    n_stapes = 14

    h0 = [None                                       for i in xrange(n_steps)]
    h1 = [tensor.zeros([n_samples, layers[1].dim_X]) for i in xrange(n_steps+1)]
    h2 = [tensor.zeros([n_samples, layers[2].dim_X]) for i in xrange(n_steps+2)]

    log_prob0 = [None for _ in xrange(n_steps)]
    log_prob1 = [None for _ in xrange(n_steps+1)]
    log_prob2 = [None for _ in xrange(n_steps+2)]

    # step 0
    log_prob0[0] = layers0[0].log_prob(h0[0])
    h1[1], log_prob1[1] = layers0[1].sample(h0[0])
    h2[2], log_prob2[2] = layers0[2].sample(h1[1])

    # step 1...N
    for step in xrange(1, n_steps):
        c0 = tensor.concatenate([            h0[step-1], h1[step+0]], axis=1)
        h0[step+0], log_prob0[step+0] = layers[0].log_prob(h0[step], c0)

        c1 = tensor.concatenate([h0[step+0], h1[step+0], h2[step+1]], axis=1)
        h1[step+1], log_prob1[step+1] = layers[1].sample(c1)

        c2 = tensor.concatenate([h1[step+1], h2[step+1]            ], axis=1)
        h2[step+2], log_prob2[step+2] = layers[2].sample(c2)

    return [h0, h1, h2], [log_prob0, log_prob1, log_prob2]

def sample_sequence(n_samples, layers0, layers):
    n_steps = 14

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
    n_steps = 14

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


        def sample_forward(x, p_layers0, p_layers, q_layers0, q_layers):
            h0 = [x[:, i, :]                                 for i in xrange(n_steps)]
            h1 = [tensor.zeros([total_size, layer_sizes[1]]) for i in xrange(n_steps+1)]
            h2 = [tensor.zeros([total_size, layer_sizes[2]]) for i in xrange(n_steps+2)]

            log_p0 = [None] * (n_steps)
            log_p1 = [None] * (n_steps+1)
            log_p2 = [None] * (n_steps+2)

            log_q0 = [None] * (n_steps)
            log_q1 = [None] * (n_steps+1)
            log_q2 = [None] * (n_steps+2)

            #--------------------------------------------------------------------------
            # draw proposal samples

            # step 0
            log_p0[0] = p_layers0[0].log_prob(h0[0])
            h1[1], log_p1[1] = p_layers0[1].sample(h0[0])
            h2[2], log_p2[2] = p_layers0[2].sample(h1[1])

            # step 1...N
            for step in xrange(1, n_steps):
                c0 = tensor.concatenate([h0[step-1], h1[step+0]            ], axis=1)
                log_p0[step] = p_layers[0].log_prob(h0[step], c0)

                c1 = tensor.concatenate([h0[step+0], h1[step+0], h2[step+1]], axis=1)
                h1[step+1], log_p1[step+1] = p_layers[1].sample(c1)

                c2 = tensor.concatenate([h1[step+1], h2[step+1]            ], axis=1)
                h2[step+2], log_p2[step+2] = p_layers[2].sample(c2)

            # Compute Q

            # step N
            log_q0[n_steps-1] = q_layers0[0].log_prob(h0[n_steps-1])
            log_q1[n_steps-1] = q_layers0[1].log_prob(h1[n_steps-1], h0[n_steps-1])
            log_q2[n_steps-1] = q_layers0[2].log_prob(h2[n_steps-1], h1[n_steps-1])

            # step 0..N-1
            for step in xrange(0, n_steps-1):
                c0 = tensor.concatenate([h0[step+1], h1[step+1]            ], axis=1)
                log_q0[step] = q_layers[0].log_prob(h0[step], c0)

                c1 = tensor.concatenate([h0[step+0], h1[step+1], h2[step+1]], axis=1)
                log_q1[step] = q_layers[1].log_prob(h1[step], c1)

                c2 = tensor.concatenate([            h1[step+1], h2[step+1]], axis=1)
                log_q2[step] = q_layers[2].log_prob(h2[step], c2)

            #import ipdb; ipdb.set_trace()

            # Remove unnecessary steps
            h1 = h1[1:-1]
            h2 = h2[2:-2]

            log_p1 = log_p1[1:-1]
            log_p2 = log_p2[2:-2]

            log_q1 = log_q1[1:-1]
            log_q2 = log_q2[2:-2]

            log_ps_joint = (sum(log_p0) + sum(log_p1) + sum(log_p2) +
                            sum(log_q0) + sum(log_q1) + sum(log_q2)) / 2

            log_proposals = sum(log_p1) + sum(log_p2)


            # Calculate sampling weights
            log_pq = log_ps_joint - log_proposals
            log_pq = log_pq.reshape([batch_size, n_samples])
            log_ps = logsumexp(log_pq, axis=1) - tensor.log(n_samples)

            w_norm = logsumexp(log_pq, axis=1)
            log_w = log_pq - tensor.shape_padright(w_norm)
            w = tensor.exp(log_w)                           # shape batch x n_samples

            w = w.reshape([batch_size*n_samples])           # shape N


            # Calculate gradients
            gradients = OrderedDict()

            # cost
            grad_cost = w * log_ps_joint
            grad_cost = -grad_cost.sum()

            params = Selector(self).get_parameters()
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(grad_cost, param, consider_constant=h0+h1+h2+[w])

            return log_ps, gradients

        #--------------------------------------------------------------------------------------------------

        batch_size = features.shape[0]
        total_size = batch_size * n_samples

        x = replicate_batch(features, n_samples)
        x = x.reshape([batch_size*n_samples, n_steps, layer_sizes[0]])

        log_ps1, gradients1 = sample_forward(x            , p_layers0, p_layers, q_layers0, q_layers)
        log_ps2, gradients2 = sample_forward(x[:, ::-1, :], q_layers0, q_layers, p_layers0, p_layers)

        log_ps = logplusexp(log_ps1, log_ps2)
        gradients = merge_gradients(gradients1, gradients2)
        # log_ps = log_ps1
        # gradients = gradients1

        return log_ps, gradients


    #==================================================================================

    def estimate_log_z(self, n_samples):
        p_layers  = self.p_layers
        p_layers0 = self.p_layers0
        q_layers  = self.q_layers
        q_layers0 = self.q_layers0
        n_layers  = len(p_layers)

        samples, log_prob_p = sample_sequence(n_samples, p_layers0, p_layers)
        log_prob_q = log_prob_sequence(rev(samples), q_layers0, q_layers)

        log_prob_p0, log_prob_p1, log_prob_p2 = log_prob_p
        log_prob_q0, log_prob_q1, log_prob_q2 = log_prob_q

        log_prob_p = sum(log_prob_p0) + sum(log_prob_p1[1:-1]) + sum(log_prob_p2[2:-2])
        log_prob_q = sum(log_prob_q0) + sum(log_prob_q1[1:-1]) + sum(log_prob_q2[2:-2])

        log_prob1  = (log_prob_q - log_prob_p) / 2

        # samples, log_prob1 = sample_sequence(n_samples, p_layers0, p_layers)
        # log_prob2 = log_prob_sequence(rev(samples), q_layers0, q_layers)
        #
        # log_prob_p = (sum(log_prob2) - sum(log_prob1)) / 2
        # log_prob_p = (log_prob2 - log_prob1) / 2

        return log_prob1, log_prob1


    # #@application(inputs=['n_samples'], outputs=['samples', 'log_p', 'log_q'])
    # def sample_p(self, n_samples):
    #     """
    #     """
    #     p_layers = self.p_layers
    #     q_layers = self.q_layers
    #     n_layers = len(p_layers)
    #
    #     samples = [None] * n_layers
    #     log_p = [None] * n_layers
    #
    #     samples[n_layers - 1], log_p[n_layers - 1] = p_layers[n_layers - 1].sample(n_samples)
    #     for l in reversed(xrange(1, n_layers)):
    #         samples[l - 1], log_p[l - 1] = p_layers[l - 1].sample(samples[l])
    #
    #     # Get log_q
    #     log_q = self.log_prob_q(samples)
    #
    #     return samples, log_p, log_q
    #
    #
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
