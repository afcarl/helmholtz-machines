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

from helmholtz import flatten_values, unflatten_values
from helmholtz.dvae import DVAE
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.prob_layers import replicate_batch, logsumexp

logger = logging.getLogger("avg-activations.py")

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

    assert isinstance(brick, DVAE)

    #----------------------------------------------------------------------
    logger.info("Compiling function...")

    x = tensor.fmatrix('features')

    prob_z = brick.q.sample_expected(x)
    #prob_zz = -tensor.log(1./prob_z - 1)
    prob_zz = prob_z
    sum_z  = prob_zz.sum(axis=0)
    sum_z2 = (prob_zz**2).sum(axis=0)

    rho = brick.p.theano_rng.uniform(
                size=prob_z.shape, 
                low=0, high=1.,
                dtype=prob_z.dtype) 

    z = tensor.switch(rho >= 1-prob_z, (rho-1)/prob_z + 1, 0.)
    x_recons = brick.p.sample_expected(z)

    do_avg_z = theano.function(
                        [x], 
                        [sum_z, sum_z2, x_recons],
                        name="do_avg_z", allow_input_downcast=True)

    #----------------------------------------------------------------------

    n_samples = tensor.iscalar('n_samples')

    z_prob = brick.p_top.sample_expected()
    rho = brick.p.theano_rng.uniform(
                size=(100, 512), 
                low=0, high=1.,
                dtype=prob_z.dtype) 

    a = (rho-1)/z_prob + 1
    xi = tensor.switch(a > 0., a, 0)

    x_sampled = brick.p.sample_expected(xi)

    do_sample = theano.function(
                        [], 
                        x_sampled,
                        name="do_sample", allow_input_downcast=True) #, on_unused_input='warn')

    #----------------------------------------------------------------------
    logger.info("Loading dataset...")
    batch_size = 100

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)

    train_stream, valid_stream, test_stream = (
            Flatten(DataStream(
                data,
                iteration_scheme=SequentialScheme(data.num_examples, batch_size)
            ), which_sources='features')
        for data, batch_size in ((data_train, batch_size),
                                 (data_valid, batch_size),
                                 (data_test, batch_size))
    )


    z_sum = None
    for data in train_stream.get_epoch_iterator():
        x = data[0]
        z_sum_batch, z_sum2_batch, x_recons = do_avg_z(x)
        if z_sum is None:
            z_sum  = z_sum_batch
            z_sum2 = z_sum2_batch
        else:
            z_sum  += z_sum_batch
            z_sum2 += z_sum2_batch

    z_sum  /= data_train.num_examples
    z_sum2 /= data_train.num_examples
    std = np.sqrt(np.abs(z_sum2-z_sum**2))


    idx = np.argsort(z_sum)
    z_sum  = z_sum[idx]
    z_sum2 = z_sum2[idx]
    std    = std[idx] 

    print(z_sum)
    print(z_sum2)
    print(std)

    import pylab
    pylab.plot(z_sum)
    pylab.plot(z_sum-std, c='b')
    pylab.plot(z_sum+std, c='b')
    pylab.fill_between(np.arange(512), z_sum-std, z_sum+std, alpha=0.5)
    pylab.title("Post-logistic avg. unit activation (DVAE: 512-512-512, learned prior)")
    pylab.xlabel("unit")
    pylab.ylabel("p(z_i = 1)")

    x = x.reshape(batch_size, 28, 28)
    x_recons = x_recons.reshape(batch_size, 28, 28)
    x_sampled = do_sample().reshape(batch_size, 28, 28)

    pylab.figure()
    pylab.gray()
    pylab.imshow(img_grid(x))

    pylab.figure()
    pylab.gray()
    pylab.imshow(img_grid(x_recons))

    pylab.figure()
    pylab.gray()
    pylab.imshow(img_grid(x_sampled))

    pylab.show(True)
    #pylab.show(True)
        #print(do_avg_z(x))
        #features = data_train.get_data(None, n)[0]
        #features = features.reshape((1, x_dim))
        #ess.append(do_ess(features, n_samples))
