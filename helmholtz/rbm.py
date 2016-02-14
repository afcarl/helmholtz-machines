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

from . import logsumexp, logplusexp
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
    def __init__(self, dim_x, dim_h=None, cd_iterations=3, **kwargs):
        super(RBMTopLayer, self).__init__(**kwargs)

        if dim_h is None:
            dim_h = dim_x

        self.dim_x = dim_x
        self.dim_h = dim_h

        self.cd_iterations = cd_iterations

    def _allocate(self):
        self.b = shared_floatx_zeros((self.dim_x,), name="b")            # visible bias
        self.c = shared_floatx_zeros((self.dim_h,), name="c")            # hidden bias
        self.W = shared_floatx_zeros((self.dim_x, self.dim_h), name="W") # weights
        self.parameters = [self.b, self.c, self.W]

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)
        self.biases_init.initialize(self.c, self.rng)
        self.weights_init.initialize(self.W, self.rng)

    @application(outputs=['X_expected'])
    def sample_expected(self, n_samples):
        """
        """
        iterations = 100

        pv = 0.5 * tensor.ones((n_samples, self.dim_x))

        rand_v = theano_rng.uniform(size=(iterations, n_samples, self.dim_x), nstreams=N_STREAMS)
        rand_h = theano_rng.uniform(size=(iterations, n_samples, self.dim_h), nstreams=N_STREAMS)

        # negative phase samples CD #k
        def step(rand_v, rand_h, pv, W, b, c):
            #v = bernoulli(pv)
            v = tensor.cast(rand_v <= pv, floatX)
            ph = sigmoid(tensor.dot(v, W) + c)
            h = tensor.cast(rand_h <= ph, floatX)
            pv = sigmoid(tensor.dot(h, W.T) + b)
            return pv

        scan_result, scan_updates = theano.scan(
                fn=step,
                sequences=[rand_v, rand_h],
                outputs_info=[pv],
                non_sequences=[self.W, self.b, self.c] )

        assert len(scan_updates) == 0
        return scan_result[-1]

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        """ Sampls *n_samples* from this model.

        Returns
        -------
        X        : tensor.fmatrix (shape n_samples x dim_x)
        log_prob : tensor.fvector (shape n_sampls)
        """
        pv = self.sample_expected(n_samples)
        v = bernoulli(pv)

        return v, self.log_prob(v)


    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        """ Evaluate the log-probability for the given X.

        Parameters
        ----------
        X:      T.tensor
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the given observed X
        """

        log_prob = tensor.sum(
                logplusexp(tensor.log(1.), tensor.dot(X, self.W) + self.c), axis=1
            ) + tensor.sum(X * self.b, axis=1)

        return log_prob


    @application(inputs=['X', 'weights'], outputs='gradients')
    def get_gradients(self, X, weights=1.):

        # gradients for the positive phase
        grads_pos = super(RBMTopLayer, self).get_gradients(X, weights)

        # negative phase samples CD #k
        v = X
        for i in xrange(self.cd_iterations):
            ph = sigmoid(tensor.dot(v, self.W) + self.c)
            h = bernoulli(ph)
            pv = sigmoid(tensor.dot(h, self.W.T) + self.b)
            v = bernoulli(pv)

        # negative phase gradients
        grads_neg = super(RBMTopLayer, self).get_gradients(v, 1.)

        grads = OrderedDict()
        for k, v in grads_pos.items():
            grads[k]  = grads_pos[k] - grads_neg[k]

        return grads


    def estimate_log_z(self, n_samples, beta=10000):
        """ Run scan-based annealed importance sampling.

        Returns
        -------
        w :  tensor.fvector  (shape: n_sample)
            return the aggretate p(v_k) / p(v_{k-1}) for n_samples
        """
        # if isinstance(beta, int):
        #     beta = numpy.linspace(0, 1, beta)

        iterations = beta.shape[0]

        rand_v = theano_rng.uniform(size=(iterations, n_samples, self.dim_x), nstreams=N_STREAMS)
        rand_h = theano_rng.uniform(size=(iterations, n_samples, self.dim_h), nstreams=N_STREAMS)

        # Initial v from factorial bernoulli
        pv = 0.5 * tensor.ones((n_samples, self.dim_x))
        v = tensor.cast(rand_v[0] <= pv, floatX)

        # Initial \omega is just - log p_0(v)
        # w = -self.dim_x * tensor.log(0.5) * tensor.ones( (n_samples,) )
        w = tensor.zeros( (n_samples,) )

        def step(beta, rand_v, rand_h, v_prev, w, W, b, c):

            # calculate log probs for old sample, current annealing distribution
            log_prob_prev = beta * self.log_prob(v_prev)

            # get next sample ...
            ph = sigmoid(beta * (tensor.dot(v_prev, W) + c))
            h  = tensor.cast(rand_h <= ph, floatX)
            pv_next = sigmoid(beta * (tensor.dot(h, W.T) + b))
            v_next = tensor.cast(rand_v <= pv_next, floatX)

            # ... and log prob for next sample, current annealing distribution
            log_prob_next = beta * self.log_prob(v_next)

            w += log_prob_prev - log_prob_next

            return v_next, w

        scan_results, scan_updates = theano.scan(
                fn=step,
                sequences=[beta, rand_v, rand_h],
                outputs_info=[v, w],
                non_sequences=[self.W, self.b, self.c]
            )

        assert len(scan_updates) == 0

        v, w = scan_results

        # Add p_K(v) to last iterations \omega obtain final w
        w = w[-1] + self.log_prob(v[-1])

        # multiply by Za
        w += (self.dim_x) * tensor.log(2)

        return w
