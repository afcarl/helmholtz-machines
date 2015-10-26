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

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

import helmholtz.datasets as datasets

from helmholtz import flatten_values, unflatten_values, replicate_batch, logsumexp
from helmholtz.dvae import DVAE
from helmholtz.bihm import BiHM
from helmholtz.rws import ReweightedWakeSleep

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

#-----------------------------------------------------------------------------

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return scale * arr

def img_grid(arr, global_scale=True):
    N, height, width = arr.shape

    rows = int(np.sqrt(N))
    cols = int(np.sqrt(N))

    if rows*cols < N:
        cols = cols + 1

    if rows*cols < N:
        rows = rows + 1

    total_height = rows * (height+1)
    total_width  = cols * (width+1)

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*(height+1), c*(width+1)
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    I = (255*I).astype(np.uint8)
    return Image.fromarray(I)

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

    assert isinstance(brick, (DVAE, ReweightedWakeSleep, BiHM))

    n_layers = len(brick.p_layers)

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    x = tensor.fmatrix('features')

    samples, log_p, log_q = brick.sample_q(x)

    sum_h = [None for l in xrange(n_layers)]
    for l, h in enumerate(samples):
        sum_h[l] = h.sum(axis=0)
        
    x_recons = brick.p_layers[0].sample_expected(samples[1])

    do_avg_z = theano.function(
                        [x], 
                        [x_recons]+sum_h,
                        name="do_avg", allow_input_downcast=True)

    #----------------------------------------------------------------------

    logger.info("Loading dataset...")
    batch_size = 100

    x_dim, train_stream, valid_stream, test_stream, = datasets.get_streams(args.data, batch_size)

    sum_h = [None for _ in xrange(n_layers)]
    for data in train_stream.get_epoch_iterator():
        x = data[0]

        res = do_avg_z(x)
        x_recons = res[0]
        sum_h_batch = res[1:]
    
        for l in xrange(n_layers):
            if sum_h[l] is None:
                sum_h[l] = sum_h_batch[l]
            else:
                sum_h[l] += sum_h_batch[l]

    sum_h = [sh / 50000. for sh in sum_h]

    import pylab
    for l in xrange(n_layers):
        sh = sum_h[l]
        idx = np.argsort(sh)
        sh  = sh[idx]

        print("---- Layer %d ----" % l)
        print(sh)

        pylab.figure()
        pylab.plot(sh)
        #pylab.plot(z_sum-std, c='b')
        #pylab.plot(z_sum+std, c='b')
        #pylab.fill_between(np.arange(512), z_sum-std, z_sum+std, alpha=0.5)
        pylab.title("Avg unit activation layer %d" % l)
        pylab.xlabel("unit")
        pylab.ylabel("p(z_i = 1)")


    #x = x.reshape(batch_size, 28, 28)
    #x_recons = x_recons.reshape(batch_size, 28, 28)
    #x_sampled = do_sample().reshape(batch_size, 28, 28)

    #pylab.figure()
    #pylab.gray()
    #pylab.imshow(img_grid(x))

    #pylab.figure()
    #pylab.gray()
    #pylab.imshow(img_grid(x_recons))

    pylab.show(True)
