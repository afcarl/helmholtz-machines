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
from blocks.serialization import load

import scipy.misc as misc

import helmholtz.datasets as datasets

from helmholtz import logsumexp
from helmholtz.biseq import BiSeq

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
            default=1000000, help="no. of samples to draw")
    parser.add_argument("--ninner", type=int,
            default=10, help="no. of samples to draw")
    parser.add_argument("--batch-size", "-bs", type=int,
            default=10000, help="no. of samples to draw")
    parser.add_argument("experiment", help="Experiment to load")
    args = parser.parse_args()

    logger.info("Loading model %s..." % args.experiment)
    with open(args.experiment, "rb") as f:
        m = load(f)

    if isinstance(m, MainLoop):
        m = m.model

    brick = m.get_top_bricks()[0]
    while len(brick.parents) > 0:
        brick = brick.parents[0]

    assert isinstance(brick, BiSeq)

    np.random.seed();
    for layer in brick.p_layers:
        layer.theano_rng.seed(np.random.randint(500))
    for layer in brick.q_layers:
        layer.theano_rng.seed(np.random.randint(500))

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    np.random.seed(999)

    batch_size = 100
    n_samples = tensor.iscalar('n_samples')
    #x = tensor.matrix('features')
    #x_ = replicate_batch(x, n_samples)

    log_z1, log_z2 = brick.estimate_log_z(n_samples)

    do_z = theano.function(
                        [n_samples],
                        [log_z1, log_z2],
                        name="do_z", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Computing Z...")

    batch_size = args.batch_size

    n_samples = []
    log_z1  = []
    log_z2 = []

    for k in xrange(0, args.nsamples, batch_size):
        psxp, psxp2 = do_z(batch_size)

        import ipdb; ipdb.set_trace()

        # psxp, psxp2 = float(psxp), float(psxp2)

        # n_samples.append(k)
        # log_psxp.append(psxp)
        # log_psxp2.append(psxp2)


    import pandas as pd
    df = pd.DataFrame({'k': n_samples, 'log_psxp': log_psxp, 'log_psxp2': log_psxp2})
    df.save("est-Z-inner%d.pkl" % (args.ninner))
