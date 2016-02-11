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

class RBMTopLayer(Initializable, ProbabilisticTopLayer):
    """ Top level RBM """
    def __init__(self, dim_x, dim_h=None, **kwargs):
        super(RBMTopLayer, self).__init__(**kwargs)

        if dim_h is None:
            dim_h = dim_x

        self.dim_x = dim_x
        self.dim_h = dim_h

    def _allocate(self):
        self.W = shared_floatx_zeros((self.dim_x, self.dim_h), name="W") # encoder weights
        self.b = shared_floatx_zeros((self.dim_x,), name="b")            # visible bias
        self.c = shared_floatx_zeros((self.dim_h,), name="c")            # hidden bias
        self.parameters = [self.b, self.c, self.W]

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)
        self.biases_init.initialize(self.c, self.rng)
        self.weights_init.initialize(self.W, self.rng)

    @application(outputs=['X_expected'])
    def sample_expected(self, n_samples):
        raise NotImplemented

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):

        raise NotImplemented
        #------------------------------------------------------------------

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

        ph = sigmoid(tensor.dot(X, self.W) + self.b)

        E = -tensor.sum(tensor.dot(X, self.W) * ph, axis=1) \
                - tensor.sum(X  * self.b, axis=1) \
                - tensor.sum(ph * self.c, axis=1)

        return E


    @application(inputs=['X', 'weights'], outputs='gradients')
    def get_gradients(self, X, weights=1.):
        grads_pos = super(RBMTopLayer, self).get_gradients(X, weights)

        # negative phase
        for i in xrange(1):
            ph = sigmoid(tensor.dot(X, self.W) + self.b)
            h = bernoulli(ph)
            pv = sigmoid(tensor.dot(X, self.W.T) + self.c)
            v = bernoulli(pv)

        grads_neg = super(RBMTopLayer, self).get_gradients(v, )

        grads = OrderedDict()
        for k, v in grads_pos.items():
            grads[k]  = grads_pos[k] - grads_neg[k]

        return grads



        # # positive phase
        # ph = sigmoid(tensor.dot(X, self.W) + self.b)
        #
        # dW = -tensor.outer(X, ph)
        # db = -X.sum(axis=0)
        # dc = -ph.sum(axis=0)
        #
        # # negative phase
        # for i in xrange(1):
        #     h = bernoulli(ph)
        #     pv = sigmoid(tensor.dot(X, self.W.T) + self.c)
        #     v = bernoulli(pv)
        #     ph = sigmoid(tensor.dot(X, self.W) + self.b)
        #
        # dW += tensor.outer(v, ph)
        # db += v.sum(axis=0)
        # dc += ph.sum(axis=0)
        #
        # grads = OrderedDict(
        #     ((self.W, dW),
        #      (self.b, db),
        #      (self.c, dc))
        # )
        #
        # import ipdb; ipdb.set_trace()
        #
        # return grads
