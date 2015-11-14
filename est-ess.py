#!/usr/bin/env python 

from __future__ import print_function, division

import sys
sys.path.append("..")
sys.setrecursionlimit(100000)

import os
import logging

import numpy as np
import cPickle as pickle

import theano
import theano.tensor as tensor

from PIL import Image
from argparse import ArgumentParser
from progressbar import ProgressBar
from scipy import stats

from blocks.main_loop import MainLoop


import helmholtz.datasets as datasets

from helmholtz import flatten_values, unflatten_values, replicate_batch, logsumexp
from helmholtz.bihm import BiHM
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--nsamples", "--samples", "-s", type=int, 
            default=100, help="no. of samples to draw")
    parser.add_argument("experiment", help="Experiment to load")
    args = parser.parse_args()

    logger.info("Loading model %s..." % args.experiment)
    with open(args.experiment, "rb") as f:
        m = pickle.load(f)

    if isinstance(m, MainLoop):
        m = m.model

    brick = m.get_top_bricks()[0]
    while len(brick.parents) > 0:
        brick = brick.parents[0]

    assert isinstance(brick, (ReweightedWakeSleep, BiHM, GMM))

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    batch_size = 1
    n_samples = tensor.iscalar('n_samples')
    x = tensor.matrix('features')

    x_ = replicate_batch(x, n_samples)
    samples, log_p, log_q = brick.sample_q(x_)

    # Reshape and sum
    samples = unflatten_values(samples, batch_size, n_samples)
    log_p = unflatten_values(log_p, batch_size, n_samples)
    log_q = unflatten_values(log_q, batch_size, n_samples)


    # Importance weights for q proposal for p
    log_p_all = sum(log_p)   # This is the python sum over a list
    log_q_all = sum(log_q)   # This is the python sum over a list

    log_pq = (log_p_all-log_q_all)-tensor.log(n_samples)
    w_norm = logsumexp(log_pq, axis=1)
    log_wp = log_pq-tensor.shape_padright(w_norm)
    wp = tensor.exp(log_wp)

    wp_ = tensor.mean(wp)
    wp2_ = tensor.mean(wp**2)

    ess_p = n_samples * (wp_**2 / wp2_)

    # Importance weights for q proposal for p*
    wps = brick.importance_weights(log_p, log_q)

    wps_ = tensor.mean(wps)
    wps2_ = tensor.mean(wps**2)

    ess_ps = n_samples * (wps_**2 / wps2_)

    do_ess = theano.function(
                        [x, n_samples], 
                        [ess_p, ess_ps],
                        name="do_ess", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)

    n_samples = 10000
    n_examples = 10000

    ess_p = []
    ess_ps = []
    for n in xrange(n_examples):
        features = data_train.get_data(None, n)[0]
        features = features.reshape((1, x_dim))
        ep, eps = do_ess(features, n_samples)
        ess_p.append(ep)
        ess_ps.append(eps)

    ess_p = np.asarray(ess_p)
    ess_ps = np.asarray(ess_ps)

    print("ESS p : %f +-%f (std: %f)" % (ess_p.mean(), stats.sem(ess_p), np.std(ess_p)))
    print("ESS p*: %f +-%f (std: %f)" % (ess_ps.mean(), stats.sem(ess_ps), np.std(ess_ps)))
