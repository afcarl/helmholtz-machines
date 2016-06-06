#!/usr/bin/env python

"""

Estimate the partition function of an RBM using Annealed Importance Sampling
(AIS) or (optionally for small models) exact marginalization.

"""

from __future__ import division, print_function

import numpy as np
import theano
import theano.tensor as tensor

from progressbar import ProgressBar
from collections import OrderedDict
from scipy.misc import logsumexp

from blocks.main_loop import MainLoop
import blocks.serialization as serialization

from helmholtz.rbm import RBMTopLayer
from helmholtz import logsumexp, logplusexp
from helmholtz.distributions import bernoulli


import dwzest.inferenceByEMC_Montreal as dwzest

#============================================================================

if __name__ == "__main__":
    import os.path
    import cPickle as pickle

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", default="rbm_model.pkl")
    parser.add_argument("--select-temp", "--stt", action="store_true")
    parser.add_argument("--st-sweeps", type=int, default=1000)
    parser.add_argument("--st-replica", type=int, default=100)
    args = parser.parse_args()

    with open(args.model, "r") as f:
        #try:
        model = serialization.load(f)
        #except Exception as e:
        #    print(e)
        #    f.seek(0)
        #    model = pickle.load(f)

    if isinstance(model, MainLoop):
        model = model.model
    model = model.get_top_bricks()[0]
    rbm = model.p_layers[-1]

    # Extract parameters
    b = rbm.b.get_value()
    c = rbm.c.get_value()
    W = rbm.W.get_value()

    b = [float(f) for f in list(b)]
    c = [float(f) for f in list(c)]
    W = [[float(f) for f in list(r)] for r in list(W)]

    inv_temp_fname = "%s-invtemp" % args.model

    if not args.select_temp:
        print("Loading temperatures from %s" % inv_temp_fname)
        with open(inv_temp_fname, "r") as f:
            inv_temp = pickle.load(f)
    else:
        print("Selecting temperatures...")
        inv_temp = dwzest.selectEMCTemperatures1_RBM_Qubo(b, c, W, seed=1, annealSweeps=args.st_sweeps, annealReplica=args.st_replica, targetER=0.5)
        with open(inv_temp_fname, "w") as f:
            pickle.dump(inv_temp, f)

    print("Running LogZ_RBM_Qubo...")
    ret1 = dwzest.calcLogZ_RBM_Qubo(b, c, W, inv_temp, seed=1, nSweepsBeyondBurnIn=1000, burnIn=None, evaluateEvery=1)
    print("Running LogZ_RBM_Qubo...")
    ret2 = dwzest.calcLogZ_RBM_Qubo(b, c, W, inv_temp, seed=2, nSweepsBeyondBurnIn=1000, burnIn=None, evaluateEvery=1)

    print("logZ (%d): %6.2f   %6.2f" % (len(inv_temp), ret1[-1], ret2[-1]))

    #import ipdb; ipdb.set_trace()
    exit(1)














    #------------------------------------------------------------------------
    print("Compiling Theano function...")

    beta = tensor.fvector('beta')
    batch_size = tensor.iscalar('batch_size')

    #wv, wh = estimate_log_z(rbm, batch_size, beta)
    w = estimate_log_z(rbm, batch_size, beta)


    do_log_z = theano.function(
                [batch_size, beta],
                w,
                allow_input_downcast=True)

    #------------------------------------------------------------------------
    print("Running AIS...")

    beta = np.linspace(0.0, 1.0, args.beta_steps)

    # beta = np.concatenate([
    #             np.linspace(0.0 , 0.5,  args.beta_steps // 3),
    #             np.linspace(0.5 , 0.7,  args.beta_steps // 3),
    #             np.linspace(0.7 , 1.0,  args.beta_steps // 3),
    #         ])

    n_samples = args.nsamples
    batch_size = 100000 // len(beta)

    wv = np.zeros((0,))
    wh = np.zeros((0,))
    pbar = ProgressBar()
    for i in pbar(xrange(n_samples // batch_size)):
        wv_ = do_log_z(batch_size, beta)
        wv = np.concatenate([wv, wv_])

        if i % 10 == 0:
            print("log Z estimate: %5.2f +- %5.2f" %
                    (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv)))) )

        # print("log Z estimate %5.2f +- %5.2f  //  %5.2f +- %5.2f" %
        #         (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv))),
        #          logsumexp(wh) - np.log(len(wh)), wh.std() / np.sqrt(np.log(len(wh))),
        #         )
        #     )
    log_zv = logsumexp(wv) - np.log(len(wv))
    # log_zh = logsumexp(wh) - np.log(len(wh))

    print("log Z estimate: %5.2f +- %5.2f" %
            (logsumexp(wv) - np.log(len(wv)), wv.std() / np.sqrt(np.log(len(wv)))) )

    fname = os.path.splitext(args.model)[0]+"-z-est.pkl"
    print("Saving AIS samples in '%s'" % fname)

    import pandas as pd
    #df = pd.DataFrame({'wv': wv, 'wh': wh}, index=np.arange(len(wv)))
    df = pd.DataFrame({'wv': wv}, index=np.arange(len(wv)))
    df.to_pickle(fname)

    log_z = log_zv

    if args.plot:
        from bokeh.plotting import figure, output_file, show

        fname = os.path.splitext(args.model)[0]+"-z-est.html"
        print("Saving plot to '%s'" % fname)
        output_file(fname)

        n_samples = range(100, len(w), 100)
        log_z = [logsumexp(w[:u]) - np.log(len(w[:u])) for u in n_samples]

        p = figure()
        p.title = "AIS (%d intermediate distributions)" % args.beta_steps
        p.xaxis.axis_label = "# independent samples"
        p.yaxis.axis_label = "estimated log Z"
        p.line(n_samples, log_z,
                line_width=2, legend="")
        if log_z_exact:
            p.line([n_samples[0], n_samples[-1]], [log_z_exact, log_z_exact],
                    line_width=2, color='red',
                    legend='exact')
        show(p)
