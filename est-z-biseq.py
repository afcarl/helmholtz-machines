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

from blocks.extensions import SimpleExtension
from blocks.main_loop import MainLoop
from blocks.serialization import load

import scipy.misc as misc

import helmholtz.datasets as datasets

from helmholtz import logsumexp
from helmholtz.biseq import BiSeq

#logger = logging.getLogger("sample.py")
logger = logging.getLogger(__name__)

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--nsamples", "--samples", "-s", type=int,
            default=100000000, help="no. of samples to draw")
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


    log_z1 = logsumexp(log_z1)
    log_z2 = logsumexp(log_z2)

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

    def calc_estimate(log_z_list):
        return misc.logsumexp(log_z_list) - np.log(batch_size * len(log_z_list))

    for k in xrange(0, args.nsamples, batch_size):
        log_z1_, log_z2_ = do_z(batch_size)

        log_z1.append(float(log_z1_))
        log_z2.append(float(log_z2_))

        if k % 1000000 == 0:
            print("log Z ~= %5.3f  (%d P samples)" % (calc_estimate(log_z1), k))
            print("log Z ~= %5.3f  (%d Q samples)" % (calc_estimate(log_z2), k))

    print("log Z ~= %5.3f  (%d P samples)" % (calc_estimate(log_z1), args.nsamples))
    print("log Z ~= %5.3f  (%d Q samples)" % (calc_estimate(log_z2), args.nsamples))

    exit(0)

    import pandas as pd
    df = pd.DataFrame({'k': n_samples, 'log_psxp': log_psxp, 'log_psxp2': log_psxp2})
    df.save("est-Z-inner%d.pkl" % (args.ninner))


#----------------------------------------------------------------------------


class EstimateLogZ(SimpleExtension):
    def __init__(self, model, n_samples=100000, batch_size=100, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("on_interrupt", True)

        self.model = model
        self.n_samples = n_samples
        self.batch_size = batch_size
        self._compile()

        super(EstimateLogZ, self).__init__(**kwargs)

    def _compile(self):
        logger.info("Compile estimate_z")

        batch_size = tensor.iscalar('batch_size')
        log_z_p, log_z_q = self.model.estimate_log_z(batch_size)
        log_z_p = logsumexp(log_z_p)
        log_z_q = logsumexp(log_z_q)

        self.do_est = theano.function(
                        [batch_size],
                        [log_z_p, log_z_q],
                        name="estimate_log_z")

    def estimate_log_z(self):
        lzest_p = []
        lzest_q = []

        logger.info("Estimating log Z started")
        for k in range(0, self.n_samples, self.batch_size):
            lzest_p_, lzest_q_ = self.do_est(self.batch_size)

            lzest_p.append(float(lzest_p_))
            lzest_q.append(float(lzest_q_))
        
        lzest_p = misc.logsumexp(lzest_p) - np.log(self.n_samples)
        lzest_q = misc.logsumexp(lzest_q) - np.log(self.n_samples)

        logger.info("Estimating log Z finished")
        return lzest_p, lzest_q
        

    def do(self, which_callback, *args):
        cur_row = self.main_loop.log.current_row

        log_z_p, log_z_q = self.estimate_log_z()

        cur_row['log_z'] = log_z_p
        cur_row['log_z_p'] = log_z_p
        cur_row['log_z_q'] = log_z_q

        # add new entries
        for key, value in cur_row.items():
            if not "_ps" in key:
                continue
            new_entry = key.replace("_ps", "_p")
            cur_row[new_entry] = value + log_z_p

