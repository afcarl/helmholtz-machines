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

from blocks.main_loop import MainLoop

from helmholtz import replicate_batch, logsumexp
from helmholtz.bihm import BiHM
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep

logger = logging.getLogger("sample.py")

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def scale_norm(arr):
    """ Scale and shoft the given array to be 0..1 """
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

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


# def sample_rws(brick, args):
#     assert isinstance(brick, ReweightedWakeSleep)
#
#     #----------------------------------------------------------------------
#     # Compile functions
#     logger.info("Compiling function...")
#
#     n_samples = tensor.iscalar('n_samples')
#
#     samples, log_p, log_q = brick.sample(n_samples)
#
#     if args.expected:
#         # Ok, take the second last and sample expected
#         x = brick.p_layers[0].sample_expected(samples[1])
#     else:
#         x = samples[0]
#
#     x = x.reshape([n_samples]+img_shape)
#
#     do_sample = theano.function(
#                         [n_samples],
#                         x,
#                         name="do_sample", allow_input_downcast=True)
#
#     #----------------------------------------------------------------------
#     logger.info("Sample from p(x, h) ...")
#
#     x_p = do_sample(args.nsamples)
#     img = img_grid(x_p, global_scale=True)
#
#     fname = os.path.splitext(args.experiment)[0]
#     fname += "-psamples.png"
#
#     logger.info("Saving %s ..." % fname)
#     img.save(fname)
#
#     if args.show:
#         import pylab
#
#         pylab.figure()
#         pylab.gray()
#         pylab.axis('off')
#         pylab.imshow(img, interpolation='nearest')
#
#         pylab.show(block=True)
#
#
# def sample_bihm(brick, args):
#     assert isinstance(brick, (BiHM, GMM))
#
#     #----------------------------------------------------------------------
#     # Compile functions
#     logger.info("Compiling function...")
#
#     n_inner = tensor.iscalar('n_inner')
#     n_samples = tensor.iscalar('n_samples')
#     oversample = tensor.iscalar('oversample')
#
#     samples, log_w = brick.sample(n_samples, oversample=oversample, n_inner=n_inner)
#
#     if args.expected:
#         # Ok, take the second last and sample expected
#         x = brick.p_layers[0].sample_expected(samples[1])
#     else:
#         x = samples[0]
#
#     x = x.reshape([n_samples]+img_shape)
#
#     do_sample = theano.function(
#                         [n_samples, oversample, n_inner],
#                         [x, log_w],
#                         name="do_sample", allow_input_downcast=True)
#
#     #----------------------------------------------------------------------
#
#     n_samples = tensor.iscalar('n_samples')
#
#     samples, _, _ = brick.sample_p(n_samples)
#
#     if args.expected:
#         # Ok, take the second last and sample expected
#         x_p = brick.p_layers[0].sample_expected(samples[1])
#     else:
#         x_p = samples[0]
#
#     x_p = x_p.reshape([n_samples]+img_shape)
#
#     do_sample_p = theano.function(
#                         [n_samples],
#                         [x_p, samples[1]],
#                         name="do_sample_p", allow_input_downcast=True)
#
#     #----------------------------------------------------------------------
#     logger.info("Sample from model...")
#
#     n_layers = len(brick.p_layers)
#     n_samples = args.nsamples
#
#     x = [None] * n_samples
#     log_w = [None] * n_samples
#     progress = ProgressBar()
#     for n in progress(xrange(n_samples)):
#         x[n], log_w[n] = do_sample(1, args.oversample, args.ninner)
#
#     x = np.concatenate(x)
#     img = img_grid(x, global_scale=True)
#
#     fname = os.path.splitext(args.experiment)[0]
#     fname += "-samples.png"
#
#     logger.info("Saving %s ..." % fname)
#     img.save(fname)
#
#     #----------------------------------------------------------------------
#     logger.info("Sample from p(x, h) ...")
#
#     x_p, h1 = do_sample_p(n_samples)
#     img_p = img_grid(x_p, global_scale=True)
#
#     fname = os.path.splitext(args.experiment)[0]
#     fname += "-psamples.png"
#
#     logger.info("Saving %s ..." % fname)
#     img_p.save(fname)
#
#     np.save("h1.npy", h1)
#
#     if args.show:
#         import pylab
#
#         pylab.figure()
#         pylab.gray()
#         pylab.axis('off')
#         pylab.imshow(img, interpolation='nearest')
#
#         pylab.figure()
#         pylab.gray()
#         pylab.axis('off')
#         pylab.imshow(img_p, interpolation='nearest')
#
#         pylab.show(block=True)


def sample_nade(brick, args):
    assert isinstance(brick, ReweightedWakeSleep)
    assert len(brick.p_layers) == 2

    #----------------------------------------------------------------------
    # Compile functions
    logger.info("Compiling function...")


    # Extract parameters
    b = brick.p_layers[0].mlp.linear_transformations[0].b
    W = brick.p_layers[0].mlp.linear_transformations[0].W

    dim_h = brick.p_layers[0].mlp.input_dim
    dim_x = brick.p_layers[0].mlp.output_dim

    n_samples = tensor.iscalar('n_samples')

    # draw samples (and throw away everything besides of top)
    samples, log_p, log_q = brick.sample(n_samples)


    def step(h, W_row, canvas):
        # h_top is shape n_samples
        # canvas is shape n_sampes x dim_X
        canvas += W_row * tensor.shape_padright(h, 1)

        return canvas


    h_top = samples[1].T                # dim_h, n_samples
    canvas = tensor.zeros((n_samples, dim_x)) + b  # n_samples, dim_x


    scan_results, scan_updates = theano.scan(step,
                                    sequences=[h_top, W],
                                    outputs_info=[canvas]
                                )

    assert len(scan_updates) == 0

    canvas = scan_results


    canvas = canvas.reshape([dim_h, n_samples]+img_shape)
    canvas = tensor.nnet.sigmoid(canvas)

    # if args.expected:
    #     # Ok, take the second last and sample expected
    #     x = brick.p_layers[0].sample_expected(samples[1])
    # else:
    #     x = samples[0]

    do_sample = theano.function(
                        [n_samples],
                        canvas,
                        name="do_sample", allow_input_downcast=True)

    #----------------------------------------------------------------------
    logger.info("Sample from p(x, h) ...")

    x_p = do_sample(args.nsamples)

    basename = os.path.splitext(args.experiment)[0]

    for i in xrange(dim_h):
        fname = basename + "-psamples%03d.png" % i

        img = img_grid(x_p[i], global_scale=True)
        logger.info("Saving %s ..." % fname)
        img.save(fname)

    import ipdb; ipdb.set_trace()
    
    if args.show:
        import pylab

        pylab.figure()
        pylab.gray()
        pylab.axis('off')
        pylab.imshow(img, interpolation='nearest')

        pylab.show(block=True)





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

    if args.shape is not None:
        img_shape = [int(i) for i in args.shape.split(',')]
    else:
        p0 = brick.p_layers[0]
        sqrt = int(np.sqrt(p0.dim_X))
        img_shape = [sqrt, sqrt]


    if isinstance(brick, ReweightedWakeSleep):
        sample_nade(brick, args)
    else:
        print("Unknown model type %s" % brick)
        exit(1)
