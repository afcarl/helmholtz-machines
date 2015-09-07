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

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

from PIL import Image
from argparse import ArgumentParser
from progressbar import ProgressBar

from blocks.main_loop import MainLoop

from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.prob_layers import replicate_batch, logsumexp

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

theano_rng = RandomStreams(seed=234)


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


def logsumexp2(a, b):
    """ Compute a numerically stable log(exp(a)+exp(b)) """
    m = tensor.maximum(a, b)
    return tensor.log(tensor.exp(a-m) + tensor.exp(b-m)) + m


def subsample(weights, n_samples):
    """ Choose *nsamples* subsamples proportionally to *weights* """
    pvals = weights.dimshuffle('x', 0).repeat(n_samples, axis=0)
    idx = theano_rng.multinomial(pvals=pvals).argmax(axis=1)
    return idx

#-----------------------------------------------------------------------------


def sample_conditional(h_upper, h_lower, p_upper, q_lower, oversample) :
    """ return (h, log_ps) """
    nsamples = 1

    h_upper = replicate_batch(h_upper, oversample)
    h_lower = replicate_batch(h_lower, oversample)

    # First, get proposals
    h1, log_1p = p_upper.sample(h_upper)
    log_1q = q_lower.log_prob(h1, h_lower)

    log_1ps = (log_1p + log_1q) / 2
    log_1 = logsumexp2(log_1p, log_1q)

    h2, log_2q = q_lower.sample(h_lower)
    log_2p = p_upper.log_prob(h2, h_upper)

    log_2ps = (log_2p + log_2q) / 2
    log_2 = logsumexp2(log_2p, log_2q)

    h_proposals = tensor.concatenate([h1, h2], axis=0)
    log_proposals = tensor.concatenate([log_1, log_2], axis=0)  # - np.log(2.)
    log_ps = tensor.concatenate([log_1ps, log_2ps], axis=0)

    # Calculate weights
    log_w = log_ps - log_proposals
    w_norm = logsumexp(log_w, axis=0)
    log_w = log_w-w_norm
    w = tensor.exp(log_w)

    idx = subsample(w, nsamples)

    return h_proposals[idx,:]


def sample_top_conditional(h_lower, p_top, q_lower):
    nsamples = 1

    # First, get proposals
    h_p, log_p = p_top.sample(nsamples)
    log_p = logsumexp2(log_p, q_lower.log_prob(h_p, h_lower))

    h_q, log_q = q_lower.sample(h_lower)
    log_q = logsumexp2(log_q, p_top.log_prob(h_p))
    
    h_proposals = tensor.concatenate([h_p, h_q], axis=0)
    log_proposals = tensor.concatenate([log_p, log_q], axis=0)  # - np.log(2.)

    # Calculate weights
    log_p = p_top.log_prob(h_proposals)
    log_q = q_lower.log_prob(h_proposals, h_lower)
    ps = (log_p + log_q) / 2
    log_w = ps - log_proposals
    w_norm = logsumexp(log_w, axis=0)
    log_w = log_w-w_norm
    w = tensor.exp(log_w)

    idx = subsample(w, nsamples)

    return h_proposals[idx, :]


def sample_bottom_conditional(h_upper, p_upper, ll_function, oversample, ninner):
    nsamples = 1

    #h_upper = replicate_batch(h_upper, oversample)

    # First, get proposals
    x = p_upper.sample_expected(h_upper)

    return x

    # Evaluate q(x)
    _, log_q = ll_function(x, ninner)

    # Calculate weights
    log_w = (log_q - log_p) / 2
    w_norm = logsumexp(log_w, axis=0)
    log_w = log_w-w_norm
    w = tensor.exp(log_w)

    idx = subsample(w, nsamples)

    return x[idx, :]


#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--expected", "-e", action="store_true",
            help="Display expected output from last layer")
    parser.add_argument("--nsamples", "--samples", "-s", type=int, 
            default=100, help="no. of samples to draw")
    parser.add_argument("--oversample", "--oversamples", type=int, 
            default=1000)
    parser.add_argument("--ninner", type=int, 
            default=100, help="no. of q(x) samples to draw")
    parser.add_argument("--shape", type=str, default=None,
            help="shape of output samples")
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

    assert isinstance(brick, (ReweightedWakeSleep, GMM))

    if args.shape is not None:
        img_shape = [int(i) for i in args.shape.split(',')]
    else:
        p0 = brick.p_layers[0]
        sqrt = int(np.sqrt(p0.dim_X))
        img_shape = [sqrt, sqrt]

    #----------------------------------------------------------------------
    # Compile functions
    logger.info("Compiling conditional-sampling functions...")

    n_layers = len(brick.p_layers)
    oversample = tensor.iscalar('oversamples')
    n_inner = tensor.iscalar('n_inner')

    do_conditionals = []

    #----------------------------------------------------------------------
    h_upper = tensor.fmatrix('h_upper')
    h = sample_bottom_conditional(
            h_upper,  
            brick.p_layers[0],
            brick.log_likelihood,
            oversample, n_inner)

    do_conditionals.append(
            theano.function(
                [h_upper], h,
                #[h_upper, oversample, n_inner], h,
                name="bottom_conditional", allow_input_downcast=True))

    #----------------------------------------------------------------------
    for l in range(1, n_layers-1):
        print("Compiling %d" % l)
        h_upper = tensor.fmatrix('h_upper')
        h_lower = tensor.fmatrix('h_lower')

        h = sample_conditional(
                    h_upper, h_lower, 
                    brick.p_layers[l],
                    brick.q_layers[l-1],
                    oversample)

        do_conditionals.append(
            theano.function(
                    [h_lower, h_upper, oversample], h,
                    name="conditional%d"%l, allow_input_downcast=True))

    #----------------------------------------------------------------------

    samples = [None] * n_layers

    samples[n_layers-1], _ = brick.p_layers[n_layers-1].sample(1)
    for l in reversed(xrange(2, n_layers)):
        samples[l-1], _ = brick.p_layers[l-1].sample(samples[l])

    if args.expected:
        # Ok, take the second last and sample expected
        samples[0] = brick.p_layers[0].sample_expected(samples[1])
    else:
        samples[0], _ = brick.p_layers[0].sample(samples[1])

    do_sample_p = theano.function(
                        [], 
                        samples,
                        name="do_sample_p", allow_input_downcast=True)

    #----------------------------------------------------------------------

    logger.info("Sample from model...")

    sweeps = 10
    n_layers = len(brick.p_layers)
    n_samples = args.nsamples
    n_inner = args.ninner
    oversample = args.oversample

    x = [None] * n_samples

    progress = ProgressBar()
    for n in progress(xrange(n_samples)):
        samples = do_sample_p()

        for _ in xrange(sweeps):
            # Upwards...
            for l in xrange(1, n_layers-1):
                samples[l] = do_conditionals[l](samples[l-1], samples[l+1], oversample)
    
            # XXX: do top level
    
            # ...downwards...
            for l in reversed(xrange(1, n_layers-1)):
                samples[l] = do_conditionals[l](samples[l-1], samples[l+1], oversample)
    
            #samples[0] = do_conditionals[0](samples[1], oversample, n_inner)
            samples[0] = do_conditionals[0](samples[1])

        x[n] = samples[0]
    
    x = np.concatenate(x)
    x = x.reshape( [n_samples,]+img_shape)
    img = img_grid(x, global_scale=True)

    fname = os.path.splitext(args.experiment)[0]
    fname += "-mcsamples.png"

    logger.info("Saving %s ..." % fname)
    img.save(fname)

    if args.show:
        import pylab

        pylab.figure()
        pylab.gray()
        pylab.axis('off')
        pylab.imshow(img, interpolation='nearest')

        pylab.figure()
        pylab.gray()
        pylab.axis('off')
        pylab.imshow(img_p, interpolation='nearest')
 
        pylab.show(block=True)
