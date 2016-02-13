#!/usr/bin/env python

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as tensor

from helmholtz.rbm import RBMTopLayer

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

    rbm = RBMTopLayer(5, 5, **inits)
    rbm.initialize()


    #--------------------------------------------------------------------------

    log_p = rbm.log_prob(v)
    grads = rbm.get_gradients(v)

    updates = OrderedDict()
    for sh_var in (rbm.W, rbm.b, rbm.c):
         updates[sh_var] = sh_var + lr * grads[sh_var]

    do_rbm = theano.function([v, lr], [log_p]+grads.values(), updates=updates, allow_input_downcast=True
            )

    #--------------------------------------------------------------------------

    n_samples = tensor.iscalar('n_samples')
    samples = rbm.sample(n_samples)
    do_sample= theano.function([n_samples], samples, allow_input_downcast=True)

    #--------------------------------------------------------------------------

    iterations = 10000
    beta = tensor.arange(iterations) / iterations

    n_samples = tensor.iscalar('n_samples')
    w = rbm.estimate_log_z(n_samples, beta)
    do_log_z = theano.function([n_samples], w, allow_input_downcast=True)

    def estimate_log_z():
        w = do_log_z(100)
        print("log Z = %f" % w.mean())

    #--------------------------------------------------------------------------

    lr = 3e-2
    v = np.array(
        [[1., 1., 1., 0., 0.],
         [0., 0., 1., 1., 1.]])

    # batch_size = 100
    # v = np.zeros( (batch_size, 10))

    for i in xrange(100000):
        log_probs, db, dc, dW = do_rbm(v, lr)

        if i % 100 == 0:
            print(log_probs.sum())
            estimate_log_z()

    samples = do_sample(20)
    print(samples)

    # import ipdb; ipdb.set_trace()
