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
from scipy.misc import logsumexp

from blocks.main_loop import MainLoop


import helmholtz.datasets as datasets

from helmholtz import flatten_values, unflatten_values
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
            default=100000, help="no. of samples to draw")
    parser.add_argument("--ninner", type=int, 
            default=1000, help="no. of samples to draw")
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

    #assert isinstance(brick, (ReweightedWakeSleep, BiHM, GMM))
    assert isinstance(brick, (BiHM, GMM))

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    np.random.seed(999)

    batch_size = 1
    n_samples = tensor.iscalar('n_samples')
    #x = tensor.matrix('features')
    #x_ = replicate_batch(x, n_samples)

    samples, log_p, log_q = brick.sample_p(1)
    log_px, log_psx = brick.log_likelihood(samples[0], n_samples)

    log_p = sum(log_p)
    log_q = sum(log_q)

    log_pxp  = log_px - log_p
    log_psxp = 1/2.*log_psx + 1/2.*(log_q-log_p)

    do_z = theano.function(
                        [n_samples], 
                        [log_pxp, log_psxp],
                        name="do_z", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Computing Z...")

    np.random.seed(); dummy = np.random.randint(50)
    for _ in xrange(dummy):
        _, _ = do_z(args.ninner)

    log_pxp = []
    log_psxp = []
    for k in xrange(args.nsamples):
        pxp, psxp = do_z(args.ninner)

        pxp = float(pxp)
        psxp = float(psxp)

        log_pxp.append(pxp)
        log_psxp.append(psxp)

        if k % 10 == 0:
            z_est = (logsumexp(log_psxp)-np.log(k+1))/2.
        
            print("[%d samples] Z (p*) estimate: %s" % (k, z_est))

    import pandas as pd
    df = pd.DataFrame({'items': log_psxp})
    df.save("est-Z-inner%d-spl%d.pkl" % (args.ninner, args.nsamples))
