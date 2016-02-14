#!/usr/bin/env python

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np

from scipy.misc import logsumexp

import theano
import theano.tensor as tensor

from helmholtz.rbm import RBMTopLayer

#===================================================================================

def logplusexp(a, b):
    m = np.maximum(a, b)
    return np.log(np.exp(a-m) + np.exp(b-m)) + m

def bitpattern(width):
    """ Renerate exaustive binary patterns as a 2d NumPy array.

    E.g., for width=2 it generates a 4 \times 2 matrix:
    [[0, 0],
     [1, 0],
     [0, 1],
     [1, 1]], dtype=np.uint8
    """
    arr = np.arange(2**width)

    pattern = np.empty( (2**width, width), dtype=np.uint8)
    for i in xrange(width):
        pattern[:, i] = (arr >> i) & 1

    return pattern

#===================================================================================

def exact_log_z(W, b, c):
    dim_x, = b.shape

    v = bitpattern(dim_x)
    E = np.sum(b * v, axis=1) + np.sum(logplusexp(np.log(1), np.dot(v, W) + c), axis=1)

    return logsumexp(E)


def exact_log_z_check(W, b, c):
    dim_x, = b.shape
    dim_h, = c.shape

    p = bitpattern(dim_x+dim_h)
    v = p[:,:dim_x]
    h = p[:,dim_x:]

    E = np.sum(np.dot(v, W) * h, axis=1) \
            +np.sum(b * v, axis=1) \
            +np.sum(c * h, axis=1)

    return logsumexp(E)

#===================================================================================

if __name__ == "__main__":
    import numpy as np
    import theano

    from blocks.initialization import Uniform, IsotropicGaussian, Constant

    inits = {
        'weights_init': IsotropicGaussian(std=0.01),
        'biases_init': Constant(-1.0),
    }

    lr = tensor.fscalar('learning_rate')
    v = tensor.fmatrix('v')

    rbm = RBMTopLayer(5, 4, cd_iterations=10, **inits)
    rbm.initialize()


    #--------------------------------------------------------------------------

    log_p = rbm.log_prob(v)
    grads = rbm.get_gradients(v)

    updates = OrderedDict()
    for sh_var in (rbm.W, rbm.b, rbm.c):
         updates[sh_var] = sh_var - lr * grads[sh_var]

    do_rbm = theano.function([v, lr], [log_p]+grads.values(), updates=updates, allow_input_downcast=True
            )

    #--------------------------------------------------------------------------

    n_samples = tensor.iscalar('n_samples')
    samples = rbm.sample(n_samples)
    do_sample= theano.function([n_samples], samples, allow_input_downcast=True)

    #--------------------------------------------------------------------------

    iterations = 10
    beta = tensor.arange(iterations) / iterations

    n_samples = tensor.iscalar('n_samples')
    w = rbm.estimate_log_z(n_samples, beta)
    do_log_z = theano.function([n_samples], w, allow_input_downcast=True)

    def estimate_log_z():
        # n_samples = 10000
        # batch_size = 1000
        # w = np.zeros((0,))
        # for i in xrange(n_samples // batch_size):
        #     w_ = do_log_z(batch_size)
        #     w = np.concatenate([w, w_])
        #     print("log Z estimate %5.2f +- %5.2f" %
        #          (logsumexp(w) - np.log((i+1)*batch_size),
        #           w.std() / np.sqrt(n_samples))
        #         )
        # log_z = logsumexp(w) - np.log(n_samples)

        #------------------------------------------------------
        W = rbm.W.get_value()
        b = rbm.b.get_value()
        c = rbm.c.get_value()

        log_z_exact = exact_log_z(W, b, c)
        # log_z_check = exact_log_z_check(W, b, c)
        # assert np.allclose(log_z_exact, log_z_check)
        #
        # print("Exact  log Z: %5.2f" % log_z_exact)
        # print("Approx log Z: %5.2f" % log_z)

        return log_z_exact

    #--------------------------------------------------------------------------

    lr = 1e-3
    v = np.array(
        [[1., 1., 0, 0., 0.],
         [0., 0., 0, 1., 1.]])

    # batch_size = 100
    # v = np.zeros( (batch_size, 10))
    print("== Estimating Z ==")
    log_z = estimate_log_z()
    print("Initial log Z: %5.2f" % log_z)

    print("== Learning ==")
    for i in xrange(100000):
        log_probs, db, dc, dW = do_rbm(v, lr)

        if i % 1000 == 0:
            log_z = estimate_log_z()
            log_probs = log_probs.mean()
            print("%5.2f - %5.2f = %5.2f" %
                    (log_probs, log_z, log_probs-log_z))

    samples = do_sample(20)
    print(samples)

    # Save model
    import cPickle as pickle

    with open("rbm_model.pkl", "w") as f:
        pickle.dump(rbm, f)
