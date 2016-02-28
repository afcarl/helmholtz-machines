#!/usr/bin/env python

"""

Estimate the partition function of an RBM using Annealed Importance Sampling
(AIS) or (optionally for small models) exact marginalization.

"""

from __future__ import division, print_function

import numpy as np
import theano
import theano.tensor as tensor

from blocks.extensions import SimpleExtension

from progressbar import ProgressBar
from collections import OrderedDict
from scipy.misc import logsumexp

from helmholtz.rbm import RBMTopLayer
from helmholtz.distributions import bernoulli, theano_rng, N_STREAMS

#----------------------------------------------------------------------------

def logplusexp(a, b):
    """ NumPy implementation of numerically stable logplusexp(a, b).

    Computes log(exp(a) + exp(b)).
    """
    m = np.maximum(a, b)
    return np.log(np.exp(a-m) + np.exp(b-m)) + m

def bitpattern(width):
    """ Renerate exaustive binary patterns as a 2d NumPy array.

    E.g., for width=2 it generates a 4 \times 2 matrix:
    [[0, 0],
     [1, 0],
     [0, 1],
     [1, 1]], dtype=np.uint8
    """
    arr = np.arange(2**width)

    pattern = np.empty( (2**width, width), dtype=np.uint8)
    for i in xrange(width):
        pattern[:, i] = (arr >> i) & 1

    return pattern

def exact_log_z(W, b, c):
    """ NumPy implementation to compute exact log Z """
    dim_x, = b.shape
    dim_h, = c.shape

    if dim_x > dim_h:
        h = np.zeros((2**16, dim_h), dtype=np.uint8)
        h[:, -min(16, dim_h):] = bitpattern(min(16, dim_h))
        E_ = np.sum(c * h, axis=1) + np.sum(logplusexp(np.log(1), np.dot(h, W.T) + b), axis=1)
        E = logsumexp(E_)
        for h_major in range(1, 2**(dim_h-16)):
            major_bits = (h_major & (1 << np.arange(dim_h-16))) > 0
            h[:, :-min(16,dim_h)] = major_bits
            E_ = np.sum(c * h, axis=1) + np.sum(logplusexp(np.log(1), np.dot(h, W.T) + b), axis=1)
            E = logplusexp(E, logsumexp(E_))
        return E
    else:
        v = bitpattern(dim_x)
        E = np.sum(b * v, axis=1) + np.sum(logplusexp(np.log(1), np.dot(v, W) + c), axis=1)

    return logsumexp(E)

#----------------------------------------------------------------------------

def estimate_log_z(rbm, n_samples, beta=10000):
    """ Run annealed importance sampling (AIS).

    Returns
    -------
    w :  tensor.fvector  (shape: n_sample)
        aggretate p(v_k) / p(v_{k-1}) for n_samples
    """
    iterations = beta.shape[0]

    dim_x = rbm.dim_x
    dim_h = rbm.dim_h

    rand_v = theano_rng.uniform(size=(iterations, n_samples, dim_x), nstreams=N_STREAMS)
    rand_h = theano_rng.uniform(size=(iterations, n_samples, dim_h), nstreams=N_STREAMS)

    # Initial v from factorial random bernoulli
    pv = 0.5 * tensor.ones((n_samples, dim_x))
    ph = 0.5 * tensor.ones((n_samples, dim_h))
    v = bernoulli(pv) #, rand_v[0])
    h = bernoulli(ph) #, rand_h[0])

    # Initial \omega is just - log p_0(v)
    # w = -self.dim_x * tensor.log(0.5) * tensor.ones( (n_samples,) )
    #wv = tensor.zeros( (n_samples,) ) - dim_h * tensor.log(2)
    #wh = tensor.zeros( (n_samples,) ) - dim_x * tensor.log(2)

    wv = -rbm.unnorm_log_prob(v, 0.)
    wh = -rbm.unnorm_log_prob_given_h(h, 0.)

    def step(beta, rand_v, rand_h, v_prev, h_prev, wv, wh, W, b, c):

        # get next sample ...
        ph_next = rbm.prob_h_given_v(v_prev, beta)
        h_next  = bernoulli(ph_next, rand_h)
        pv_next = rbm.prob_v_given_h(h_next, beta)
        v_next  = bernoulli(pv_next, rand_v)

        log_prob_v_prev = rbm.unnorm_log_prob(v_prev, beta)
        log_prob_v_next = rbm.unnorm_log_prob(v_next, beta)

        log_prob_h_next = rbm.unnorm_log_prob_given_h(h_next, beta)
        log_prob_h_prev = rbm.unnorm_log_prob_given_h(h_prev, beta)

        wv += log_prob_v_prev - log_prob_v_next
        wh += log_prob_h_prev - log_prob_h_next

        return v_next, h_next, wv, wh

    scan_results, scan_updates = theano.scan(
            fn=step,
            sequences=[beta, rand_v, rand_h],
            outputs_info=[v, h, wv, wh],
            non_sequences=[rbm.W, rbm.b, rbm.c]
        )

    assert len(scan_updates) == 0

    v, h, wv, wh = scan_results

    # Add p_K(v) to last iterations \omega obtain final w
    wv = wv[-1] + rbm.unnorm_log_prob(v[-1])
    wh = wh[-1] + rbm.unnorm_log_prob_given_h(h[-1])

    # multiply by Za
    wv += (dim_h + dim_x) * tensor.log(2)
    wh += (dim_h + dim_x) * tensor.log(2)

    return wv, wh

#----------------------------------------------------------------------------

class ComputeLogZ(SimpleExtension):
    def __init__(self, rbm, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("on_interrupt", True)

        self.rbm = rbm
        super(ComputeLogZ, self).__init__(**kwargs)

    def compute_log_z(self):
        rbm = self.rbm
        W, b, c = (p.get_value() for p in (rbm.W, rbm.b, rbm.c))

        print("** CALCULATING log Z ***")
        log_z = exact_log_z(W, b, c)

        return log_z

    def do(self, which_callback, *args):
        cur_row = self.main_loop.log.current_row

        logZ = self.compute_log_z()

        cur_row['log_Z'] = logZ

        # add new entries
        for key, value in cur_row.items():
            if not "_unnorm_" in key:
                continue
            new_entry = key.replace("_unnorm_", "_")
            cur_row[new_entry] = value + logZ

#============================================================================

if __name__ == "__main__":
    import os.path
    import cPickle as pickle

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", default="rbm_model.pkl")
    parser.add_argument("--plot", action="store_true", help="Enabple matplotlib plotting")
    parser.add_argument("--exact", action="store_true", help="Calculate exact log Z")
    parser.add_argument("--nsamples", type=int, default=100000)
    parser.add_argument("--beta-steps", type=int, default=10000)
    args = parser.parse_args()

    with open(args.model, "r") as f:
        model = pickle.load(f)

    rbm = model.get_top_bricks()[0]

    # Always calculate exact log Z if dim_x <= 15
    if rbm.dim_x <= 15 or args.exact:
        print("Computing exact log Z...")
        W = rbm.W.get_value()
        b = rbm.b.get_value()
        c = rbm.c.get_value()

        log_z_exact = exact_log_z(W, b, c)
        print("log Z = %5.2f" % log_z_exact)
    else:
        log_z_exact = None

    #------------------------------------------------------------------------
    print("Compiling Theano function...")

    beta = tensor.fvector('beta')
    batch_size = tensor.iscalar('batch_size')

    wv, wh = estimate_log_z(rbm, batch_size, beta)

    do_log_z = theano.function(
                [batch_size, beta],
                wv,
                allow_input_downcast=True)

    #------------------------------------------------------------------------
    print("Running AIS...")

    beta = np.concatenate([
#                np.linspace(0.0 , 0.5 ,  2000),
#                np.linspace(0.5 , 0.7 , 10000),
#                np.linspace(0.7 , 1.0 ,  2000)

                np.linspace(0.0, 1.0, 18000)
            ])

    n_samples = args.nsamples
    batch_size = 100000 // len(beta)

    wv = np.zeros((0,))
    wh = np.zeros((0,))
    pbar = ProgressBar()
    for i in pbar(xrange(n_samples // batch_size)):
        wv_ = do_log_z(batch_size, beta)
        #wv_, wh_ = do_log_z(batch_size, beta_steps)
        # wh = np.concatenate([wh, wh_])
        wv = np.concatenate([wv, wv_])

        if i % 10 == 0:
            print("log Z estimate: %5.2f +- %5.2f" %
                    (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv)))) )

        # print("log Z estimate %5.2f +- %5.2f  //  %5.2f +- %5.2f" %
        #         (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv))),
        #          logsumexp(wh) - np.log(len(wh)), wh.std() / np.sqrt(np.log(len(wh))),
        #         )
        #     )
    log_zv = logsumexp(wv) - np.log(len(wv))
    # log_zh = logsumexp(wh) - np.log(len(wh))

    print("log Z estimate: %5.2f +- %5.2f" %
            (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv)))) )

    fname = os.path.splitext(args.model)[0]+"-z-est.pkl"
    print("Saving AIS samples in '%s'" % fname)

    import pandas as pd
    #df = pd.DataFrame({'wv': wv, 'wh': wh}, index=np.arange(len(wv)))
    df = pd.DataFrame({'wv': wv}, index=np.arange(len(wv)))
    df.to_pickle(fname)

    log_z = log_zv

    if args.plot:
        from bokeh.plotting import figure, output_file, show

        fname = os.path.splitext(args.model)[0]+"-z-est.html"
        print("Saving plot to '%s'" % fname)
        output_file(fname)

        n_samples = range(100, len(w), 100)
        log_z = [logsumexp(w[:u]) - np.log(len(w[:u])) for u in n_samples]

        p = figure()
        p.title = "AIS (%d intermediate distributions)" % args.beta_steps
        p.xaxis.axis_label = "# independent samples"
        p.yaxis.axis_label = "estimated log Z"
        p.line(n_samples, log_z,
                line_width=2, legend="")
        if log_z_exact:
            p.line([n_samples[0], n_samples[-1]], [log_z_exact, log_z_exact],
                    line_width=2, color='red',
                    legend='exact')
        show(p)
