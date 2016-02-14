#!/usr/bin/env python

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np

from scipy.misc import logsumexp

import theano
import theano.tensor as tensor

from helmholtz.rbm import RBMTopLayer

#----------------------------------------------------------------------------

def logplusexp(a, b):
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
    dim_x, = b.shape

    v = bitpattern(dim_x)
    E = np.sum(b * v, axis=1) + np.sum(logplusexp(np.log(1), np.dot(v, W) + c), axis=1)

    return logsumexp(E)

#===================================================================================

if __name__ == "__main__":
    import cPickle as pickle

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", default="rbm_model.pkl")
    parser.add_argument("--exact", action="store_true", help="Calculate exact log Z")
    parser.add_argument("--nsamples", type=int, default=1000000)
    parser.add_argument("--beta-steps", type=int, default=1000)
    args = parser.parse_args()

    with open(args.model, "r") as f:
        rbm = pickle.load(f)

    # Always calculate exact log Z if dim_x <= 15
    if rbm.dim_x <= 15 or args.exact:
        print("Computing excapt log Z...")
        W = rbm.W.get_value()
        b = rbm.b.get_value()
        c = rbm.c.get_value()

        log_z_exact = exact_log_z(W, b, c)
        print("log Z = %5.2f" % log_z_exact)
    else:
        log_z_exact = None

    #------------------------------------------------------------------------

    beta_steps = tensor.iscalar('beta_steps')
    batch_size = tensor.iscalar('batch_size')

    beta = tensor.arange(beta_steps) / beta_steps
    w = rbm.estimate_log_z(batch_size, beta)

    do_log_z = theano.function([batch_size, beta_steps], w, allow_input_downcast=True)

    #------------------------------------------------------------------------
    n_samples = args.nsamples
    beta_steps = args.beta_steps
    batch_size = 1000000 // beta_steps

    w = np.zeros((0,))
    for i in xrange(n_samples // batch_size):
        w_ = do_log_z(batch_size, beta_steps)
        w = np.concatenate([w, w_])
        print("log Z estimate %5.2f +- %5.2f" %
                (logsumexp(w) - np.log(len(w)),
                 w.std() / np.sqrt(np.log(len(w)))
                )
            )
    log_z = logsumexp(w) - np.log(n_samples)
