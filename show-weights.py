#!/usr/bin/env python 

from __future__ import print_function, division

import sys
sys.path.append("..")
sys.setrecursionlimit(100000)

import os
import logging
import pylab

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

from helmholtz import flatten_values, unflatten_values
from helmholtz.bihm import BiHM
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.prob_layers import replicate_batch, logsumexp

from sample import img_grid


logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


def logistic(x):
    return 1. / (1+np.exp(-x))


#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser("Estimate the effective sample size")
    parser.add_argument("experiment", help="Experiment to load")
    parser.add_argument("--shape", type=str, default=None,
            help="shape of output samples")
    parser.add_argument("--symmetric", action="store_true", 
            help="Display weights with symmetric colormap. (0.0 == gray)")
    parser.add_argument("--global-scale", "--global", action="store_true", 
            help="Global colormap scaleing.")
    parser.add_argument("--sigmoid", action="store_true", 
            help="Map weights through a logistic sigmoid function.")
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


    # Image shape
    if args.shape is not None:
        img_shape = [int(i) for i in args.shape.split(',')]
    else:
        p0 = brick.p_layers[0]
        sqrt = int(np.sqrt(p0.dim_X))
        img_shape = [sqrt, sqrt]

    #----------------------------------------------------------------------

    bp = brick.p_layers[0].mlp.linear_transformations[-1].b.get_value()
    Wp = brick.p_layers[0].mlp.linear_transformations[-1].W.get_value()

    bq = brick.q_layers[0].mlp.linear_transformations[0].b.get_value()
    Wq = brick.q_layers[0].mlp.linear_transformations[0].W.get_value().T

    Wp = Wp 

    bp = bp.reshape(img_shape)
    Wp = Wp.reshape([-1,]+img_shape)
    Wq = Wq.reshape([-1,]+img_shape)

    dim_p = Wp.shape[0]
    dim_q = Wq.shape[0]

    assert dim_p == dim_q

    if args.sigmoid:
        Wp = logistic(Wp)
        Wq = logistic(Wq)

    pylab.figure()
    pylab.imshow(img_grid(Wp, args.global_scale, args.symmetric))
    pylab.gray()
    pylab.title("P weights")

    pylab.figure()
    pylab.imshow(img_grid(Wp, args.global_scale, args.symmetric))
    pylab.gray()
    pylab.title("Q weights")

    pylab.show(block=True)
