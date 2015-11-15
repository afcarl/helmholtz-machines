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
#from blocks.monitoring.eavluators import 


import helmholtz.datasets as datasets

from helmholtz import flatten_values, unflatten_values, replicate_batch, logsumexp
from helmholtz.bihm import BiHM
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.vae import VAE

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
            default=1000, help="no. of samples")
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

    assert isinstance(brick, (ReweightedWakeSleep, BiHM, GMM, VAE))

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    n_layers = len(brick.p_layers)
    n_samples = tensor.iscalar('n_samples')
    x = tensor.matrix('features')
    batch_size = x.shape[0]

    x_ = replicate_batch(x, n_samples)
    samples, log_p, log_q = brick.sample_q(x_)

    # Reshape and sum
    samples = unflatten_values(samples, batch_size, n_samples)
    log_p = unflatten_values(log_p, batch_size, n_samples)
    log_q = unflatten_values(log_q, batch_size, n_samples)



    # Importance weights for q proposal for p
    log_p_all = sum(log_p)   # This is the python sum over a list
    log_q_all = sum(log_q)   # This is the python sum over a list

    log_pq = (log_p_all-log_q_all)
    log_px = logsumexp(log_pq, axis=1) - tensor.log(n_samples)

    log_qp = (log_q_all-log_p_all)
    log_kl = tensor.sum(log_qp, axis=1) / n_samples

    total_kl = log_kl + log_px
    layer_kl = [tensor.sum(lq-lp, axis=1) / n_samples for lp, lq in zip(log_p[:], log_q[:])]

    do_kl = theano.function(
                        [x, n_samples], 
                        [log_px, total_kl]+layer_kl,
                        name="do_kl", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")

    n_samples = args.nsamples
    batch_size = 5*max(1, 10000 // args.nsamples)
    
    x_dim, stream_train, stream_valid, stream_test = datasets.get_streams(args.data, batch_size)
    stream = stream_test

    log_px = np.array([])
    total_kl = np.array([])
    layer_kl = [np.array([]) for _ in xrange(n_layers)]
    for batch in stream.get_epoch_iterator():
        features = batch[0]
        ret = do_kl(features, n_samples)
        log_px_batch, total_kl_batch = ret[:2]
        layer_kl_batch = ret[2:]

        log_px = np.concatenate([log_px, log_px_batch])
        total_kl = np.concatenate([total_kl, total_kl_batch])
        for l, kl_batch in enumerate(layer_kl_batch):
            layer_kl[l] = np.concatenate([layer_kl[l], kl_batch])

    print(total_kl.shape)
    print(log_px.shape)

    print("log p(x): %f +-%f (std: %f)" % (log_px.mean(), stats.sem(log_px), np.std(log_px)))
    print("KL(q|p) : %f +-%f (std: %f)" % (total_kl.mean(), stats.sem(total_kl), np.std(total_kl)))
    print()
    for l, kl in enumerate(layer_kl):
        print("KL%d(q|p) : %f +-%f (std: %f)" % (l, kl.mean(), stats.sem(kl), np.std(kl)))
        

    #print("ESS p : %f +-%f (std: %f)" % (ess_p.mean(), stats.sem(ess_p), np.std(ess_p)))
    #print("ESS p*: %f +-%f (std: %f)" % (ess_ps.mean(), stats.sem(ess_ps), np.std(ess_ps)))
