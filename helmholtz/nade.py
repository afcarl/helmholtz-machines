from __future__ import division, print_function

import logging

import numpy
import theano

from collections import OrderedDict
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor import nnet

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.roles import add_role, PARAMETER, WEIGHT, BIAS
from blocks.bricks import Random, MLP, Initializable
from blocks.utils import pack, shared_floatx_zeros
from blocks.select import Selector

from .distributions import bernoulli
from .prob_layers import ProbabilisticTopLayer, ProbabilisticLayer

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

theano_rng = MRG_RandomStreams(seed=2341)
N_STREAMS = 2048


sigmoid_fringe = 1e-6

def sigmoid(val):
    return nnet.sigmoid(val).clip(sigmoid_fringe, 1.-sigmoid_fringe)

#------------------------------------------------------------------------------

class NADETopLayer(Initializable, ProbabilisticTopLayer):
    """ Top Level NADE """
    def __init__(self, dim_X, dim_H=None, **kwargs):
        super(NADETopLayer, self).__init__(**kwargs)

        if dim_H is None:
            dim_H = 2 * dim_X

        self.dim_X = dim_X
        self.dim_H = dim_H

    def _allocate(self):
        self.b = shared_floatx_zeros((self.dim_X,), name="b") # visible bias
        self.c = shared_floatx_zeros((self.dim_H,), name="c") # hidden bias
        self.W = shared_floatx_zeros((self.dim_X, self.dim_H), name="W") # encoder weights
        self.V = shared_floatx_zeros((self.dim_H, self.dim_X), name="W") # encoder weights
        self.parameters = [self.b, self.c, self.W, self.V]

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)
        self.biases_init.initialize(self.c, self.rng)
        self.weights_init.initialize(self.W, self.rng)
        self.weights_init.initialize(self.V, self.rng)

    @application(outputs=['X_expected'])
    def sample_expected(self, n_samples):
        raise NotImplemented

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        n_X, n_hid = self.dim_X, self.dim_H
        b, c, W, V = self.b, self.c, self.W, self.V

        #------------------------------------------------------------------

        a_init    = tensor.zeros([n_samples, n_hid]) + tensor.shape_padleft(c)
        post_init = tensor.zeros([n_samples], dtype=floatX)
        vis_init  = tensor.zeros([n_samples], dtype=floatX)
        rand      = theano_rng.uniform(size=(n_X, n_samples), nstreams=N_STREAMS)

        def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
            hid  = sigmoid(a)
            pi   = sigmoid(tensor.dot(hid, Vi) + bi)
            vis_i = tensor.cast(rand_i <= pi, floatX)
            post  = post + tensor.log(pi*vis_i + (1-pi)*(1-vis_i))
            a     = a + tensor.outer(vis_i, Wi)
            return a, vis_i, post

        [a, vis, post], updates = theano.scan(
                    fn=one_iter,
                    sequences=[W, V.T, b, rand],
                    outputs_info=[a_init, vis_init, post_init],
                )
        assert len(updates) == 0
        return vis.T, post[-1,:]


    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, n_hid = self.dim_X, self.dim_H
        b, c, W, V = self.b, self.c, self.W, self.V

        batch_size = X.shape[0]
        vis = X

        #------------------------------------------------------------------

        a_init    = tensor.zeros([batch_size, n_hid]) + tensor.shape_padleft(c)
        post_init = tensor.zeros([batch_size], dtype=floatX)

        def one_iter(vis_i, Wi, Vi, bi, a, post):
            hid  = sigmoid(a)
            pi   = sigmoid(tensor.dot(hid, Vi) + bi)
            post = post + tensor.log(pi*vis_i + (1-pi)*(1-vis_i))
            a    = a + tensor.outer(vis_i, Wi)
            return a, post

        [a, post], updates = theano.scan(
                    fn=one_iter,
                    sequences=[vis.T, W, V.T, b],
                    outputs_info=[a_init, post_init],
                )
        assert len(updates) == 0
        return post[-1,:]


#----------------------------------------------------------------------------

# class NADELayer(Initializable, ProbabilisticLayer):
#     """ Top Level NADE """
#     def __init__(self, dim_X, dim_Y, dim_H=None, **kwargs):
#         super(NADELayer, self).__init__(**kwargs)
#
#         if dim_H is None:
#             dim_H = (dim_X + dim_Y) / 2
#
#         self.dim_X = dim_X
#         self.dim_Y = dim_Y
#         self.dim_H = dim_H
#
#
#     def _allocate(self):
#         self.b  = shared_floatx_zeros((self.dim_X,), name="d") # visible bias
#         self.c  = shared_floatx_zeros((self.dim_H,), name="c") # hidden bias
#         self.W  = shared_floatx_zeros((self.dim_X, self.dim_H), name="W")  # encoder weights
#         self.V  = shared_floatx_zeros((self.dim_H, self.dim_X), name="V")  # decoder weights
#         self.Ub = shared_floatx_zeros((self.dim_Y, self.dim_X), name="Ub") # weights
#         self.Uc = shared_floatx_zeros((self.dim_Y, self.dim_H), name="Uc") # weights
#         self.children = [self.b, self.c, self.W, self.V, self.Ub, self.Uc]
#
#
#     def _initialize(self):
#         self.biases_init.initialize(self.b, self.rng)
#         self.biases_init.initialize(self.c, self.rng)
#         self.weights_init.initialize(self.W, self.rng)
#         self.weights_init.initialize(self.V, self.rng)
#         self.weights_init.initialize(self.Ub, self.rng)
#         self.weights_init.initialize(self.Uc, self.rng)
#
#
#     @application(outputs=['X_expected'])
#     def sample_expected(self, n_samples):
#         raise "not really!"
#
#
#     @application(outputs=['X', 'log_prob'])
#     def sample(self, Y):
#         n_X, n_hid = self.dim_X, self.dim_H
#         W, V = self.W, self.V
#
#         #------------------------------------------------------------------
#         b_cond = self.b + T.dot(Y, self.Ub)    # shape (batch, n_vis)
#         c_cond = self.c + T.dot(Y, self.Uc)    # shape (batch, n_hid)
#
#         a_init    = tensor.zeros([n_samples, n_hid]) + tensor.shape_padleft(c_cond)
#         post_init = tensor.zeros([n_samples], dtype=floatX)
#         vis_init  = tensor.zeros([n_samples], dtype=floatX)
#         rand      = self.rng.uniform((n_X, n_samples), nstreams=512)
#
#         def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
#             hid  = self.sigmoid(a)
#             pi   = self.sigmoid(tensor.dot(hid, Vi) + bi)
#             vis_i = tensor.cast(rand_i <= pi, floatX)
#             post  = post + tensor.log(pi*vis_i + (1-pi)*(1-vis_i))
#             a     = a + tensor.outer(vis_i, Wi)
#             return a, vis_i, post
#
#         [a, vis, post], updates = theano.scan(
#                     fn=one_iter,
#                     sequences=[W, V.T, b_cond, rand],
#                     outputs_info=[a_init, vis_init, post_init],
#                 )
#         assert len(updates) == 0
#         return vis.T, post[-1,:]
#
#
#     @application(inputs=['X', 'Y'], outputs='log_prob')
#     def log_prob(self, X, Y):
#         """ Evaluate the log-probability for the given samples.
#
#         Parameters
#         ----------
#         X:      T.tensor
#             samples from X
#         Y:      T.tensor
#             samples from Y
#
#         Returns
#         -------
#         log_p:  T.tensor
#             log-probabilities for the samples in X
#         """
#         n_X, n_hid = self.dim_X, self.dim_H
#         W, V = self.W, self.V
#
#         #------------------------------------------------------------------
#         b_cond = self.b + T.dot(Y, self.Ub)    # shape (batch, n_vis)
#         c_cond = self.c + T.dot(Y, self.Uc)    # shape (batch, n_hid)
#
#         batch_size = X.shape[0]
#         vis = X
#
#         #------------------------------------------------------------------
#
#         a_init    = tensor.zeros([batch_size, n_hid]) + tensor.shape_padleft(c_cond)
#         post_init = tensor.zeros([batch_size], dtype=floatX)
#
#         def one_iter(vis_i, Wi, Vi, bi, a, post):
#             hid  = self.sigmoid(a)
#             pi   = self.sigmoid(tensor.dot(hid, Vi) + bi)
#             post = post + tensor.log(pi*vis_i + (1-pi)*(1-vis_i))
#             a    = a + tensor.outer(vis_i, Wi)
#             return a, post
#
#         [a, post], updates = unrolled_scan(
#                     fn=one_iter,
#                     sequences=[vis.T, W, V.T, b_cond],
#                     outputs_info=[a_init, post_init],
#                     unroll=self.unroll_scan
#                 )
#         assert len(updates) == 0
#         return post[-1,:]
#
#
#
#         n_X, n_Y, n_hid    = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
#         b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
#
#         batch_size = X.shape[0]
#         vis = X
#         cond = Y
#
#         #------------------------------------------------------------------
#         b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
#         c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
#
#         a_init    = c_cond
#         post_init = T.zeros([batch_size], dtype=floatX)
#
#         def one_iter(vis_i, Wi, Vi, bi, a, post):
#             hid  = self.sigmoid(a)
#             pi   = self.sigmoid(T.dot(hid, Vi) + bi)
#             post = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
#             a    = a + T.outer(vis_i, Wi)
#             return a, post
#
#         [a, post], updates = unrolled_scan(
#                     fn=one_iter,
#                     sequences=[vis.T, W, V.T, b_cond.T],
#                     outputs_info=[a_init, post_init],
#                     unroll=self.unroll_scan
#                 )
#         assert len(updates) == 0
#         return post[-1,:]
#
#     def sample(self, Y):
#         """ Evaluate the log-probability for the given samples.
#
#         Parameters
#         ----------
#         Y:      T.tensor
#             samples from the upper layer
#
#         Returns
#         -------
#         X:      T.tensor
#             samples from the lower layer
#         log_p:  T.tensor
#             log-probabilities for the samples in X and Y
#         """
#         n_X, n_Y, n_hid = self.get_hyper_params(['n_X', 'n_Y', 'n_hid'])
#         b, c, W, V, Ub, Uc = self.get_model_params(['b', 'c', 'W', 'V', 'Ub', 'Uc'])
#
#         batch_size = Y.shape[0]
#         cond = Y
#
#         #------------------------------------------------------------------
#         b_cond = b + T.dot(cond, Ub)    # shape (batch, n_vis)
#         c_cond = c + T.dot(cond, Uc)    # shape (batch, n_hid)
#
#         a_init    = c_cond
#         post_init = T.zeros([batch_size], dtype=floatX)
#         vis_init  = T.zeros([batch_size], dtype=floatX)
#         rand      = theano_rng.uniform((n_X, batch_size), nstreams=512)
#
#         def one_iter(Wi, Vi, bi, rand_i, a, vis_i, post):
#             hid  = self.sigmoid(a)
#             pi   = self.sigmoid(T.dot(hid, Vi) + bi)
#             vis_i = T.cast(rand_i <= pi, floatX)
#             post  = post + T.log(pi*vis_i + (1-pi)*(1-vis_i))
#             a     = a + T.outer(vis_i, Wi)
#             return a, vis_i, post
#
#         [a, vis, post], updates = unrolled_scan(
#                     fn=one_iter,
#                     sequences=[W, V.T, b_cond.T, rand],
#                     outputs_info=[a_init, vis_init, post_init],
#                     unroll=self.unroll_scan
#                 )
#         assert len(updates) == 0
#         return vis.T, post[-1,:]
