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

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

from blocks.main_loop import MainLoop

import helmholtz.datasets as datasets

from helmholtz import replicate_batch
from helmholtz.gmm import GMM
from helmholtz.bihm import BiHM
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
    parser.add_argument("--zsamples", type=int, default=1000000,
            help="Estimate Z using this number of samples")
    parser.add_argument("--est-z", "-z", action="store_true", default=False,
            help="Estimate log Z and print estimated p* nll")
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

    assert isinstance(brick, (ReweightedWakeSleep, GMM, BiHM))

    #----------------------------------------------------------------------
    if args.est_z:
        logger.info("Estimating Z...")

        # compile theano function
        bs = tensor.iscalar('bs')
        log_z2 = brick.estimate_log_z2(bs)

        do_z = theano.function(
            [bs],
            log_z2,
            name="do_z", allow_input_downcast=True)

        #-------------------------------------------------------

        batch_size = 10000

        seq = []
        for _ in xrange(0, args.zsamples, batch_size):
            seq.append(float(do_z(batch_size)))
        
        log_z2 = logsumexp(seq) - np.log(args.zsamples)
                
        print("2 log Z ~= %5.3f" % log_z2)

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    batch_size = 1
    n_samples = tensor.iscalar('n_samples')
    x = tensor.matrix('features')

    log_p, log_ps = brick.log_likelihood(x, n_samples)
    
    #log_p = tensor.mean(log_p)
    #log_ps = tensor.mean(log_ps)

    do_nll = theano.function(
                        [x, n_samples], 
                        [log_p, log_ps],
                        name="do_nll", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)

    batch_size = 1
    num_examples = data_test.num_examples
    stream = Flatten(DataStream(
                        data_test,
                        iteration_scheme=ShuffledScheme(num_examples, batch_size)
                    ), which_sources='features')

    #n_samples = (1, 10, 100, 1000, 10000)
    n_samples = (1000,)

    dict_p = {}
    dict_ps = {}
    
    for K in n_samples:
        log_p = np.asarray([])
        log_ps = np.asarray([])
        for batch in stream.get_epoch_iterator(as_dict=True):
            log_p_, log_ps_ = do_nll(batch['features'], K)
    
            log_p = np.concatenate((log_p, log_p_))
            log_ps = np.concatenate((log_ps, log_ps_))
    
        log_p_ = stats.sem(log_p)
        log_p = np.mean(log_p)
        log_ps_ = stats.sem(log_ps)
        log_ps = np.mean(log_ps)

        dict_p[K] = log_p
        dict_ps[K] = log_ps
    
        if args.est_z:
            print("log p  /  p~  /  p*:  %5.2f+-%4.2f  /  %5.2f+-%4.2f  /  %5.2f" % (log_p, log_p_, log_ps, log_ps_, log_ps-log_z2))
        else:
            print("log p  /  p~:  %5.2f+-%4.2f  /  %5.2f+-%4.2f" % (log_p, log_p_, log_ps, log_ps_))

    print(dict_p)
    print(dict_ps)
