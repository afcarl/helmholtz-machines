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

import helmholtz.datasets as datasets

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

from PIL import Image
from argparse import ArgumentParser
from progressbar import ProgressBar

from blocks.main_loop import MainLoop

from helmholtz import replicate_batch, logsumexp
from helmholtz.bihm import BiHM
from helmholtz.rws import ReweightedWakeSleep

from sample import img_grid

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

theano_rng = RandomStreams(seed=234)

#-----------------------------------------------------------------------------

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


def sample_conditional(h_upper, h_lower, p_upper, p_lower, q_upper, q_lower, oversample) :
    """ return (h, log_ps) """
    nsamples = 1

    h_upper = replicate_batch(h_upper, oversample)
    h_lower = replicate_batch(h_lower, oversample)

    # First, get proposals
    h1, log_1pu = p_upper.sample(h_upper)
    log_1pl = p_lower.log_prob(h_lower, h1)
    log_1qu = q_upper.log_prob(h_upper, h1)
    log_1ql = q_lower.log_prob(h1, h_lower)

    log_1ps = (log_1pu + log_1pl + log_1ql + log_1qu) / 2
    log_1 = logsumexp2(log_1pu, log_1ql)

    h2, log_2ql = q_lower.sample(h_lower)
    log_2qu = q_upper.log_prob(h_upper, h2)
    log_2pl = p_lower.log_prob(h_lower, h2)
    log_2pu = p_upper.log_prob(h2, h_upper)

    log_2ps = (log_2pu + log_2pl + log_2ql + log_2qu) / 2
    log_2 = logsumexp2(log_2pu, log_2ql)

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


def sample_top_conditional(h_lower, p_top, q_lower, oversample):
    nsamples = 1

    h_lower = replicate_batch(h_lower, oversample)

    # First, get proposals
    h1, log_1p = p_top.sample(oversample)
    log_1q = q_lower.log_prob(h1, h_lower)

    log_1ps = (log_1p + log_1q) / 2
    log_1 = logsumexp2(log_1p, log_1q)

    h2, log_2q = q_lower.sample(h_lower)
    log_2p = p_top.log_prob(h2)

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


def sample_bottom_conditional(h_upper, p_upper, ll_function, q_upper, oversample, ninner):
    nsamples = 1

    """
    #h_upper = replicate_batch(h_upper, oversample)

    # First, get proposals
    x = p_upper.sample_expected(h_upper)

    return x
    """

    h_upper = replicate_batch(h_upper, oversample)
    x, log_p = p_upper.sample(h_upper)

    # Evaluate q(x) and q(h1|x)
    _, log_ql = ll_function(x, ninner)
    log_qu = q_upper.log_prob(h_upper, x)

    # Calculate weights
    log_w = (log_ql + log_qu - log_p) / 2
    w_norm = logsumexp(log_w, axis=0)
    log_w = log_w-w_norm
    w = tensor.exp(log_w)

    idx = subsample(w, nsamples)

    return x[idx, :]


def inpaint_bottom_conditional(h_upper, p_upper, ll_function, q_upper, x, mask, oversample, ninner):
    nsamples = 1

    x = replicate_batch(x, oversample)
    mask = replicate_batch(mask, oversample)
    h_upper = replicate_batch(h_upper, oversample)

    x_, _ = p_upper.sample(h_upper)
    x = mask*x + (1-mask)*x_

    log_p = p_upper.log_prob(x, h_upper)

    # Evaluate q(x)
    _, log_ql = ll_function(x, ninner)
    log_qu = q_upper.log_prob(h_upper, x)

    # Calculate weights
    log_w = (log_ql + log_qu - log_p) / 2
    w_norm = logsumexp(log_w, axis=0)
    log_w = log_w-w_norm
    w = tensor.exp(log_w)

    idx = subsample(w, nsamples)

    return x[idx, :]




#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--savepdf", "--pdf", action="store_true")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--expected", "-e", action="store_true",
            help="Display expected output from last layer")
    parser.add_argument("--nsamples", "--samples", "-s", type=int, 
            default=100, help="no. of samples to draw")
    parser.add_argument("--oversample", "--oversamples", type=int, 
            default=1000)
    parser.add_argument("--ninner", type=int, 
            default=10, help="no. of q(x) samples to draw")
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

    assert isinstance(brick, (ReweightedWakeSleep, BiHM))

    if args.shape is not None:
        img_shape = [int(i) for i in args.shape.split(',')]
    else:
        p0 = brick.p_layers[0]
        sqrt = int(np.sqrt(p0.dim_X))
        img_shape = [sqrt, sqrt]

    #----------------------------------------------------------------------
    # Compile functions
    n_layers = len(brick.p_layers)

    oversample = tensor.iscalar('oversamples')
    n_samples = tensor.iscalar('n_samples')
    n_inner = tensor.iscalar('n_inner')
    x = tensor.fmatrix('x')
    mask = tensor.fmatrix('mask')
    n_iter = 500

    #----------------------------------------------------------------------
    logger.info("Compiling even/odd-sampling...")

    def one_iter(*h):
        assert len(h) == n_layers
        h = list(h)

    def one_iter(*h):
        assert len(h) == n_layers
        h = list(h)

        for first in (1, 0):
            for l in xrange(first, n_layers, 2):
                if l == 0:
                    h[l] = inpaint_bottom_conditional(
                        h[1],
                        brick.p_layers[0],
                        brick.log_likelihood,
                        brick.q_layers[0],
                        x, mask,
                        oversample, n_inner)
                elif l == n_layers-1:
                    h[l] = sample_top_conditional(
                        h[l-1],  
                        brick.p_layers[-1],
                        brick.q_layers[-1],
                        oversample)
                else:
                    h[l] = sample_conditional(
                        h[l+1], h[l-1], 
                        brick.p_layers[l],
                        brick.p_layers[l-1],
                        brick.q_layers[l],
                        brick.q_layers[l-1],
                        oversample)
        return h

    #h, _, _ = brick.sample_q(x)
    h, _, _ = brick.sample_p(1)
    h = list(h)

    #h[0], _ = brick.p_layers[0].sample(h[1])
    #h[0] = mask * x + (1-mask) * h[0]

    outputs, updates = theano.scan(fn=one_iter, 
            outputs_info=h, 
            n_steps=n_iter)

    h = list(outputs)

    if args.expected:
        h[0] = brick.p_layers[0].sample_expected(h[1])
    
    do_evenodd = theano.function(
                    [x, mask, oversample, n_inner], h,
                    updates=updates,
                    name="evenodd", allow_input_downcast=True, on_unused_input='ignore')
        
    #----------------------------------------------------------------------
    # XXX call it XXX

    n_samples = args.nsamples
    n_inner = args.ninner
    oversample = args.oversample


    logger.info("Loading dataset...")

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)


    masks = np.zeros((4, 28, 28))
    masks[0, :14, :] = 1
    masks[1, :, :14] = 1
    masks[2, 14:, :] = 1
    masks[3, :, 14:] = 1
    masks = masks.reshape(4, 28*28)

    n_examples = 100
    x = [None] * n_examples

    #sel = [0, 1, 2, 6, 12, 15, 16, 17, 18, 20, 22]
    sel = [20, 32, 54, 15, 57, 6, 2, 1, 0, 111] #] 102, 111, 543, 234, 111] #0, 1, 2, 6, 12, 15, 16, 17, 18, 20, 22]


    progress = ProgressBar()
    for n in progress(xrange(n_examples)):
        features = data_train.get_data(None, sel[n // 10])[0]
        features = features.reshape((1, x_dim))

        noise = 0.5*np.ones_like(features)
        mask = masks[(n // 10) % 4, :].reshape(1, -1)
        features = mask * features + (1-mask)*noise
        
        h = do_evenodd(features, mask, oversample, n_inner)
        x[n] = h[0]
        x[n][0,:] = features
        x[n][:,:] = mask*features + (1-mask)*x[n]

    x = np.concatenate(x)
    x = x.reshape( [n_examples,n_iter]+img_shape)

    import pylab
    for i in xrange(0, n_iter, 2):
        fname = os.path.splitext(args.experiment)[0]
        fname += "-inpaint%03d.png" % (i // 2)

        logger.info("Saving %s ..." % fname)
        img = img_grid(x[:,i], global_scale=True)
        img.save(fname)

        if args.savepdf and (i % 10 == 0):
            fname = os.path.splitext(args.experiment)[0]
            fname += "-inpaint%03d.pdf" % i

            logger.info("Saving %s ..." % fname)
            pylab.figure()
            for j in xrange(n_samples):
                pylab.subplot(10, 10, j+1)
                pylab.gray()
                pylab.axis('off')
                pylab.imshow(x[j,i], interpolation='nearest')
            pylab.savefig(fname)
 


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
