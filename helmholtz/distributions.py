#!t usr/bin/env python 

from __future__ import division, print_function

import logging

import theano
import theano.tensor as tensor
import theano.tensor as T

import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.opt import register_canonicalize, register_specialize
from theano.gradient import grad_undefined
from theano import gof

logger = logging.getLogger(__name__)

N_STREAMS = 20148
floatX = theano.config.floatX
theano_rng = MRG_RandomStreams(seed=2341)

#-----------------------------------------------------------------------------

class NaturalGaussianOp(theano.Op):
    pass


class BernoulliOp(theano.Op):
    """ back-propable' Bernoulli distribution.
    """
    #__props__ = ('theano_rng')

    def __init__(self):
        super(BernoulliOp, self).__init__()

    def make_node(self, prob, rng=None, nstreams=None):
        assert hasattr(self, '_props')

        if rng is None:
            rng = theano_rng
        if nstreams is None:
            nstreams = N_STREAMS

        prob = theano.tensor.as_tensor_variable(prob)
        noise = rng.uniform(size=prob.shape, nstreams=nstreams)

        return theano.Apply(self, [prob, noise], [prob.type()])

    def perform(self, node, inputs, output_storage):
        logger.warning("BernoulliOp.perform(...) called")
        
        prob = inputs[0]
        noise = inputs[1]

        samples = output_storage[0]
        samples[0] = (noise < prob).astype(floatX)

    def grad(self, inputs, grads):
        logger.warning("BernoulliOp.grad(...) called")

        prob = inputs[0]
        noise = inputs[1]
        #import ipdb; ipdb.set_trace()

        #g0 = prob.zeros_like().astype(theano.config.floatX)
        g0 = prob * grads[0]
        g1 = grad_undefined(self, 1, noise)
        return [g0, g1]

bernoulli = BernoulliOp()

#-----------------------------------------------------------------------------
# Optimization

@register_canonicalize
@register_specialize
@gof.local_optimizer([BernoulliOp])
def replace_bernoulli_op(node):
    if not isinstance(node.op, BernoulliOp):
        return False

    prob = node.inputs[0]
    noise = node.inputs[1]
    
    samples = (noise < prob).astype(floatX)
    
    return [samples]


class Multinomial(theano.Op):
    '''Converts samples from a uniform into sample from a multinomial.'''
    def __init__(self, odtype):
        self.odtype = odtype

    def __eq__(self, other):
        return type(self) == type(other) and self.odtype == other.odtype

    def __hash__(self):
        return hash((type(self), self.odtype))

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, pvals,   rng = theano_rng, nstreams=None):
  
        nb_outcomes = pvals.shape[0]
        unis = rng.uniform(size =  (1,nb_outcomes) , low=0.0, high=1.0 )#np.random.uniform(size =  (1,pvals.shape[1]) , low=0.0, high=1.0 )
        pvals = tensor.as_tensor_variable(pvals)
        
        unis = tensor.as_tensor_variable(unis[0])
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = tensor.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return theano.Apply(self, [pvals, unis], [out])

    def grad(self, ins, outgrads):
        pvals, unis = ins
        (gz,) = outgrads
        return [tensor.zeros_like(x) for x in ins]

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs
        if self.odtype == 'auto':
            t = "PyArray_TYPE((PyArrayObject*) py_%(pvals)s)" % locals()
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }
        if (PyArray_DIMS(%(unis)s)[0] != PyArray_DIMS(%(pvals)s)[0])
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_ZEROS(2,
                PyArray_DIMS(%(pvals)s),
                %(t)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        { // NESTED SCOPE
        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        //
        // For each multinomial, loop over each possible outcome
        //
        for (int n = 0; n < nb_multi; ++n)
        {
            int waiting = 1;
            dtype_%(pvals)s cummul = 0.;
            const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, n);
            for (int m = 0; m < nb_outcomes; ++m)
            {
                dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                cummul += *pvals_nm;
                if (waiting && (cummul > *unis_n))
                {
                    *z_nm = 1.;
                    waiting = 0;
                }
                else
                {
                    // if we re-used old z pointer, we have to clear it out.
                    *z_nm = 0.;
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        (pvals, unis) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0]:
            raise ValueError("unis.shape[0] != pvals.shape[0]",
                             unis.shape[0], pvals.shape[0])
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = numpy.zeros(pvals.shape, dtype=node.outputs[0].dtype)

        nb_multi = pvals.shape[0]
        nb_outcomes = pvals.shape[1]

        # For each multinomial, loop over each possible outcome
        for n in range(nb_multi):
            waiting = True
            cummul = 0
            unis_n = unis[n]

            for m in range(nb_outcomes):
                cummul += pvals[n, m]
                if (waiting and (cummul > unis_n)):
                    z[0][n, m] = 1
                    waiting = False
                else:
                    z[0][n, m] = 0

#=============================================================================

if __name__ == "__main__":
    
    n_samples = tensor.iscalar("n_samples")
    prob = tensor.vector('prob')
    target_prob = tensor.vector('target_prob')

    shape = (n_samples, prob.shape[0])
    bprob = tensor.ones(shape) * prob

    samples = bernoulli(bprob, rng=theano_rng)
    
    mean = tensor.mean(samples, axis=0)
    cost = tensor.sum((mean-target_prob)**2)
    
    grads = theano.grad(cost, prob)

    print("-"*78)
    print(theano.printing.debugprint(samples))
    print("-"*78)

    do_sample = theano.function(
                inputs=[prob, target_prob, n_samples], 
                outputs=[samples, grads],
                allow_input_downcast=True, name="do_sample")

    
    #-------------------------------------------------------------------------
    n_samples = 10000
    prob = np.linspace(0, 1, 10)
    target_prob = prob

    samples, grads = do_sample(prob, target_prob, n_samples)
    print("== samples =========")
    print(samples)
    print("== mean ============")
    print(np.mean(samples, axis=0))
    print("== grads ===========")
    print(grads)
