#!/usr/bin/env python

"""
Sample from an RBM using (annealed) MCMC.
"""

from __future__ import division, print_function

from collections import OrderedDict

import logging
import numpy as np

import theano
import theano.tensor as tensor

from progressbar import ProgressBar
from scipy.misc import logsumexp
from PIL import Image

from helmholtz.distributions import bernoulli
from helmholtz.rbm import RBMTopLayer
from helmholtz.rws import ReweightedWakeSleep

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

#----------------------------------------------------------------------------

def scale_norm(arr, symmetric):
    """ Scale and shoft the given array to be 0..1 """
    if symmetric:
        upper = max(abs(arr.min()), abs(arr.max()))
        lower = -upper
    else:
        lower = arr.min()
        upper = arr.max()

    arr = arr - lower
    scale = (upper - lower)
    return arr / scale

def img_grid(arr, global_scale=True, symmetric=False):
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
        arr = scale_norm(arr, symmetric)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i], symmetric)

        offset_y, offset_x = r*(height+1), c*(width+1)
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this

    I = (255*I).astype(np.uint8)
    return Image.fromarray(I)


#============================================================================

if __name__ == "__main__":
    import os.path
    import cPickle as pickle

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", default="rbm_model.pkl")
    parser.add_argument("--expected", "-e", action="store_true",
            help="Display expected output from last layer")
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--mcmc-steps", type=int, default=10000)
    parser.add_argument("--shape", type=str, default=None,
            help="shape of output samples")
    args = parser.parse_args()

    with open(args.model, "r") as f:
        model = pickle.load(f)

    rws = model.get_top_bricks()[0]
    assert isinstance(rws, ReweightedWakeSleep)

    rbm = rws.p_layers[-1]
    assert isinstance(rbm, RBMTopLayer)

    #------------------------------------------------------------------------
    # Check if there are negative PCD samples?

    if hasattr(rbm, 'pcd') and rbm.pcd is not None:
        pcd_samples = rbm.pcd.get_value()
        n_pcd_samples = pcd_samples.shape[0]

        logger.info("Found %d negative PCD samples... Compiling... " % n_pcd_samples)

        h_top = tensor.fmatrix("samples")


        p_layers = rws.p_layers
        q_layers = rws.q_layers
        n_layers = len(p_layers)

        h = [None] * n_layers
        h[-1] = h_top
        for i, layer in reversed(list(enumerate(rws.p_layers[:-1]))):
            h[i], _ = layer.sample(h[i+1])

        if args.expected:
            h[0] = p_layers[0].sample_expected(h[1])

        do_downward = theano.function([h_top], h[0], allow_input_downcast=True)
        #--------------------------------------------------------------------

        pcd_samples = do_downward(pcd_samples[:args.nsamples])

        if args.shape is not None:
            img_shape = [int(i) for i in args.shape.split(',')]
        else:
            dim_x = pcd_samples.shape[-1]
            sqrt = int(np.sqrt(dim_x))
            img_shape = [sqrt, sqrt]

        pcd_samples = pcd_samples.reshape( [n_pcd_samples]+img_shape)
        img = img_grid(pcd_samples, global_scale=True)

        fname = os.path.splitext(args.model)[0]+"-pcdsamples.png"
        logger.info("Saving samples in '%s'" % fname)
        img.save(fname)

    #------------------------------------------------------------------------
    logger.info("Compiling Theano function...")

    batch_size = tensor.iscalar('batch_size')
    mcmc_steps = tensor.iscalar('mcmc_steps')


    samples = rbm.sample(batch_size, mcmc_steps=mcmc_steps)
    for p in reversed(rws.p_layers[1:-1]):
        samples, _ = p.sample(samples)

    if args.expected:
        samples = rws.p_layers[0].sample_expected(samples)
    else:
        samples, _ = rws.p_layers[0].sample(samples)

    do_samples = theano.function([batch_size, mcmc_steps], samples)

    #------------------------------------------------------------------------
    logger.info("Running...")

    batch_size = 5

    samples = []
    pbar = ProgressBar()
    for i in pbar(xrange(args.nsamples // batch_size)):
        samples.append(do_samples(batch_size, args.mcmc_steps))
    samples = np.concatenate(samples, axis=0)

    if args.shape is not None:
        img_shape = [int(i) for i in args.shape.split(',')]
    else:
        dim_x = samples.shape[-1]
        sqrt = int(np.sqrt(dim_x))
        img_shape = [sqrt, sqrt]

    samples = samples.reshape( [args.nsamples]+img_shape)
    img = img_grid(samples, global_scale=True)

    fname = os.path.splitext(args.model)[0]+"-samples.png"
    print("Saving samples in '%s'" % fname)
    img.save(fname)
