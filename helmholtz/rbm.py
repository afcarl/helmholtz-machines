from __future__ import division, print_function

import logging

import numpy
import theano

import scipy.misc as misc

from collections import OrderedDict
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor import nnet

from blocks.bricks import Random, MLP, Initializable
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.extensions import SimpleExtension
from blocks.roles import add_role, PARAMETER, WEIGHT, BIAS
from blocks.select import Selector
from blocks.utils import pack, shared_floatx_zeros

from . import logsumexp, logplusexp
from .distributions import bernoulli
from .prob_layers import ProbabilisticTopLayer, ProbabilisticLayer

import dwzest.inferenceByEMC_Montreal as dwzest

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
    def __init__(self, dim_x, dim_h=None, cd_iterations=5, pcd_training=False, persistent_samples=100, persistent_iterations=5, **kwargs):
        super(RBMTopLayer, self).__init__(**kwargs)

        if dim_h is None:
            dim_h = dim_x

        self.dim_x = dim_x
        self.dim_h = dim_h

        self.cd_iterations = cd_iterations
        self.pcd_training = pcd_training
        self.persistent_iterations = persistent_iterations
        self.persistent_samples = persistent_samples

    def _allocate(self):
        self.b = shared_floatx_zeros((self.dim_x,), name="b")            # visible bias
        self.c = shared_floatx_zeros((self.dim_h,), name="c")            # hidden bias
        self.W = shared_floatx_zeros((self.dim_x, self.dim_h), name="W") # weights
        self.parameters = [self.b, self.c, self.W]

        if self.persistent_samples is not None:
            self.persistent_samples = shared_floatx_zeros((self.persistent_samples, self.dim_x), name='persistent_samples')

    def _initialize(self):
        self.biases_init.initialize(self.b, self.rng)
        self.biases_init.initialize(self.c, self.rng)
        self.weights_init.initialize(self.W, self.rng)

    @application(inputs=['h', 'beta'], outputs=['pv'])
    def prob_v_given_h(self, h, beta=1.):
        """ Return p(v | h) -- with optional ennealing parameter beta """
        return sigmoid(beta * (tensor.dot(h, self.W.T) + self.b))

    @application(inputs=['v', 'beta'], outputs=['ph'])
    def prob_h_given_v(self, v, beta=1.):
        """ Return p(h | v) -- with optional annealing parameter beta """
        return sigmoid(beta * (tensor.dot(v, self.W) + self.c))

    @application(inputs=['v', 'h'], outputs='E')
    def energy(self, v, h):
        """ Rrturn the energy E = -vWh -vb - hc

        Parameters
        ----------
        v : tensor.fmatrix
        h : tensor.fmatriv

        Returns
        -------
        E : tensor.fvector
        """
        return -tensor.sum(h * tensor(v, self.W), axis=1) \
                -tensor.sum(v * self.b, axis=1) \
                -tensor.sum(h * self.c, axis=1)

    @application(inputs=['n_samples', 'mcmc_steps'], outputs=['X_expected'])
    def sample_expected(self, n_samples, mcmc_steps=10000):
        """ Run an MCMC chain to sample from the RBM.

        Returns
        -------

        v : tensor.fmatix (n_samples x dim_x)
        """

        # annealed sampling
        def step(beta, rand_v, rand_h, pv, W, b, c):
            v = bernoulli(pv, noise=rand_v)
            ph = self.prob_h_given_v(v, beta)
            h = bernoulli(ph, noise=rand_h)
            pv = self.prob_v_given_h(h, beta)
            return pv

        pv = 0.5 * tensor.ones((n_samples, self.dim_x))
        beta = 0.9 * tensor.arange(mcmc_steps) / mcmc_steps

        beta = tensor.cast(beta, 'float32')
        rand_v = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_x), nstreams=N_STREAMS)
        rand_h = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_h), nstreams=N_STREAMS)

        scan_result, scan_updates = theano.scan(
                fn=step,
                sequences=[beta, rand_v, rand_h],
                outputs_info=[pv],
                non_sequences=[self.W, self.b, self.c] )
        assert len(scan_updates) == 0
        pv = scan_result[-1]

        beta = 0.9 + 0.1 * tensor.arange(mcmc_steps) / mcmc_steps

        beta = tensor.cast(beta, 'float32')
        rand_v = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_x), nstreams=N_STREAMS)
        rand_h = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_h), nstreams=N_STREAMS)

        scan_result, scan_updates = theano.scan(
                fn=step,
                sequences=[beta, rand_v, rand_h],
                outputs_info=[pv],
                non_sequences=[self.W, self.b, self.c] )
        assert len(scan_updates) == 0
        pv = scan_result[-1]

        beta = tensor.ones((mcmc_steps,))

        rand_v = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_x), nstreams=N_STREAMS)
        rand_h = theano_rng.uniform(size=(mcmc_steps, n_samples, self.dim_h), nstreams=N_STREAMS)

        scan_result, scan_updates = theano.scan(
                fn=step,
                sequences=[beta, rand_v, rand_h],
                outputs_info=[pv],
                non_sequences=[self.W, self.b, self.c] )
        assert len(scan_updates) == 0
        pv = scan_result[-1]

        return pv

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples, mcmc_steps=10000):
        """ Sampls *n_samples* from this model.

        Returns
        -------
        v  : tensor.fmatrix (n_samples x dim_x)
        """
        pv = self.sample_expected(n_samples, mcmc_steps)

        # Sample accoring to pv
        v = bernoulli(pv)

        return v

    @application(inputs=['h', 'beta'], outputs='unnorm_log_prob')
    def unnorm_log_prob_given_h(self, h, beta=1.):
        """ Return the energy E for the given h.

        Parameters
        ----------
        h:      T.tensor

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the given observed h
        """
        log_prob = tensor.sum(
                logplusexp(tensor.log(1.), beta*(tensor.dot(h, self.W.T) + self.b)), axis=1
            ) + beta * tensor.sum(h * self.c, axis=1)

        return log_prob

    @application(inputs=['v', 'beta'], outputs='unnorm_log_prob')
    def unnorm_log_prob(self, v, beta=1.):
        """ Return the energy E for the given X.

        Parameters
        ----------
        v:      T.tensor

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the given observed X
        """
        log_prob = tensor.sum(
                logplusexp(tensor.log(1.), beta*(tensor.dot(v, self.W) + self.c)), axis=1
            ) + beta * tensor.sum(v * self.b, axis=1)

        return log_prob


    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        """ Evaluate the *UNNORMALIZED* log-probability for the given X.

        Parameters
        ----------
        X:      T.tensor
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the given observed X
        """

        return self.unnorm_log_prob(X)

    @application(inputs=['v', 'weights'], outputs=['gradients', 'recons_xentropy'])
    def get_gradients(self, v, weights=1.):
        """ Return gradients and reconstruction cross entropy to monitor progress.
        """

        # Calculate gradients
        def grads(v):
            cost = -self.unnorm_log_prob(v).mean()

            return OrderedDict((
                (self.b, theano.grad(cost, self.b, consider_constant=[v])),
                (self.c, theano.grad(cost, self.c, consider_constant=[v])),
                (self.W, theano.grad(cost, self.W, consider_constant=[v])),
            ))


        # gradients for the positive phase
        grads_pos = grads(v)

        # CD training?
        if not self.pcd_training:
            for _ in xrange(self.cd_iterations):
                ph = self.prob_h_given_v(v)
                h = bernoulli(ph)
                pv = self.prob_v_given_h(h)
                v = bernoulli(pv)
            grads_neg = grads(v)

        # Advance negative samples
        if self.persistent_samples is not None:
            # negative phase samples
            v = self.persistent_samples
            for i in xrange(self.persistent_iterations):
                ph = self.prob_h_given_v(v)
                h = bernoulli(ph)
                pv = self.prob_v_given_h(h)
                v = bernoulli(pv)

            self.pcd_updates = [(self.persistent_samples, v)]
            if self.pcd_training:
                grads_neg = grads(v)
        else:
            self.pcd_updates = []


        #unnorm_log_prob = -self.unnorm_log_prob(v).mean()
        #recons_xentropy = tensor.sum(v * tensor.log(pv) + (1-v) * tensor.log(1-pv), axis=1)
        #recons_xentropy.name = "recons_xentropy"

        # Merge positive and negative gradients
        grads = OrderedDict()
        for k, v in grads_pos.items():
            grads[k]  = grads_pos[k] - grads_neg[k]

        return grads

#-------------------------------------------------------------------------------


class EstimateLogZ(SimpleExtension):
    def __init__(self, rbm, n_samples=100, st_sweeps=100, st_replica=100, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("on_interrupt", True)

        self.rbm = rbm
        self.n_samples = n_samples
        self.st_sweeps = st_sweeps
        self.st_replica = st_replica

        self.inv_temp_old = [f/10 for f in range(11)]

        super(EstimateLogZ, self).__init__(**kwargs)

    def estimate_log_z(self):
        b = self.rbm.b.get_value()
        c = self.rbm.c.get_value()
        W = self.rbm.W.get_value()

        b = [float(f) for f in list(b)]
        c = [float(f) for f in list(c)]
        W = [[float(f) for f in list(r)] for r in list(W)]

        logger.info("Estimating log Z: using old schedule")
        seed = numpy.random.randint(1, 255)
        ret = dwzest.calcLogZ_RBM_Qubo(
                            b, c, W,
                            self.inv_temp_old,
                            seed=seed,
                            nSweepsBeyondBurnIn=self.n_samples,
                            burnIn=None,
                            evaluateEvery=1)
        log_z_old = ret[-1]

        logger.info("Estimating log Z: selecting temnperatures")
        seed = numpy.random.randint(1, 255)
        inv_temp = dwzest.selectEMCTemperatures1_RBM_Qubo(
                            b, c, W,
                            seed=seed,
                            annealSweeps=self.st_sweeps,
                            annealReplica=self.st_replica,
                            targetER=0.5)
        self.inv_temp = inv_temp

        logger.info("Estimating log Z: using old schedule")
        seed = numpy.random.randint(1, 255)
        ret = dwzest.calcLogZ_RBM_Qubo(
                            b, c, W,
                            inv_temp,
                            seed=seed,
                            nSweepsBeyondBurnIn=self.n_samples,
                            burnIn=None,
                            evaluateEvery=1)
        log_z = ret[-1]

        return log_z, log_z_old


    def do(self, which_callback, *args):
        cur_row = self.main_loop.log.current_row

        log_z, log_z_old = self.estimate_log_z()

        cur_row['log_z'] = log_z
        cur_row['log_z_old'] = log_z_old

        # add new entries
        for key, value in cur_row.items():
            if not "_unnorm" in key:
                continue
            new_entry = key.replace("_unnorm", "")
            cur_row[new_entry] = value + log_z
