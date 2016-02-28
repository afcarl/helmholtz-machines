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

from progressbar import ProgressBar
from scipy import stats
from scipy.misc import logsumexp

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

from blocks.main_loop import MainLoop

import helmholtz.datasets as datasets
from helmholtz.rbm import RBMTopLayer
est_z = __import__("est-z-rbm")

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--max-batch", type=int,
            default="10000", help="Maximum internal batch size (default: 10000)")
    parser.add_argument("--no-z-est", "-noz", action="store_true", default=False,
            help="Do not estimate log Z")
    parser.add_argument("--exact-z", action="store_true", default=False,
            help="Estimate Z using this number of samples")
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

    rws = brick
    rbm = rws.p_layers[-1]
    assert isinstance(rbm, (RBMTopLayer,))

    """
    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    x = tensor.matrix('features')

    log_pt = brick.unnorm_log_prob(x)

    do_nll = theano.function(
    [x],
    [log_pt],
    name="do_nll", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")

    x_dim, _, _, data_test = datasets.get_data(args.data)
    num_examples = data_test.num_examples

    batch_size = max(args.max_batch, 1)
    x_dim, _, _, stream = datasets.get_streams(args.data, batch_size)

    logger.info("Computing unnornalized log_prob...")
    log_pt = np.asarray([])
    for batch in stream.get_epoch_iterator(as_dict=True):
        log_pt_,  = do_nll(batch['features'])
        log_pt = np.concatenate((log_pt, log_pt_))

    log_pt = log_pt.mean()
    logger.info(" -> log pt(x)=%5.2f", log_pt)
    """

    #----------------------------------------------------------------------
    exact_z = (rbm.dim_h <= 22 or args.exact_z) and not args.no_z_est
    ais_z = not exact_z and not args.no_z_est

    if exact_z:
        logger.info("Calculating exact log z...")
        W, b, c = (p.get_value() for p in (rbm.W, rbm.b, rbm.c))
        est_z = est_z.exact_log_z(W, b, c)

        logger.info("Exact log Z = %5.2f" % float(est_z))
        logger.info("Average log p(x) = %5.2f" % float(log_pt - est_z))

    if ais_z:
        logger.info("Using AIS to estimate log z...")



    #
    #
    #
    # #----------------------------------------------------------------------
    # # print final result
    #
    # if estimate_z:
    #     print("log p / log p~ / log p* [%6d spls]:  %5.2f+-%4.2f  /  %5.2f+-%4.2f  /  %5.2f" %
    #     (K, log_p, log_p_, log_ps, log_ps_, log_ps-log_z2))
    # else:
    #     print("log p / log p~ [%6d spls]:  %5.2f+-%4.2f  /  %5.2f+-%4.2f" %
    #     (K, log_p, log_p_, log_ps, log_ps_))
