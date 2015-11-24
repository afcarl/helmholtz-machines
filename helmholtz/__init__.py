
from __future__ import division, print_function 

import sys
sys.path.append("../")


import logging
import numpy
import re
import theano

from abc import ABCMeta, abstractmethod
from theano import tensor
from collections import OrderedDict

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import Random, Initializable, MLP, Tanh, Logistic
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse, Identity
from blocks.select import Selector
from blocks.roles import PARAMETER

from initialization import RWSInitialization
from prob_layers import BernoulliTopLayer, BernoulliLayer, MultinomialTopLayer_label, MultinomialLayer

logger = logging.getLogger(__name__)
floatX = theano.config.floatX


def logsumexp(A, axis=None):
    """Numerically stable log( sum( exp(A) ) ) """
    A_max = tensor.max(A, axis=axis, keepdims=True)
    B = tensor.log(tensor.sum(tensor.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = tensor.sum(B, axis=axis)
    return B


def replicate_batch(A, repeat):
    """Extend the given 2d Tensor by repeating reach line *repeat* times.

    With A.shape == (rows, cols), this function will return an array with
    shape (rows*repeat, cols).

    Parameters
    ----------
    A : T.tensor
        Each row of this 2d-Tensor will be replicated *repeat* times
    repeat : int

    Returns
    -------
    B : T.tensor
    """
    A_ = A.dimshuffle((0, 'x', 1))
    A_ = A_ + tensor.zeros((A.shape[0], repeat, A.shape[1]), dtype=floatX)
    A_ = A_.reshape( [A_.shape[0]*repeat, A.shape[1]] )
    return A_


def flatten_values(vals, size):
    """ Flatten a list of Theano tensors.
    
    Flatten each Theano tensor in *vals* such that each of them is 
    reshaped from shape (a, b, *c) to (size, *c). In other words:
    The first two dimension of each tensor in *vals* are replaced 
    with a single dimension is size *size*.

    Parameters
    ----------
    vals : list
        List of Theano tensors

    size : int
        New size of the first dimension 
    
    Returns
    -------
    flattened_vals : list
        Reshaped version of each *vals* tensor.
    """
    return vals.reshape((size, tensor.prod(vals.shape)//size))
    # data_dim = vals[0].ndim - 2
    # assert all([v.ndim == data_dim+2 for v in vals])
    # 
    # if data_dim == 0:
    #     return [v.reshape([size]) for v in vals]
    # elif data_dim == 1:
    #     return [v.reshape([size, v.shape[2]]) for v in vals]
    # raise 

def unflatten_values(vals, batch_size, n_samples):
    """ Reshape a list of Theano tensors. 

    Parameters
    ----------
    vals : list
        List of Theano tensors
    batch_size : int
        New first dimension 
    n_samples : int
        New second dimension
    
    Returns
    -------
    reshaped_vals : list
        Reshaped version of each *vals* tensor.
    """
    data_dim = vals[0].ndim - 1
    assert all([v.ndim == data_dim+1 for v in vals])

    if data_dim == 0:
        return [v.reshape([batch_size, n_samples]) for v in vals]
    elif data_dim == 1:
        return [v.reshape([batch_size, n_samples, v.shape[1]]) for v in vals]
    raise 


def merge_gradients(old_gradients, new_gradients, scale=1.):
    """Take and merge multiple ordered dicts 
    """
    if isinstance(new_gradients, (dict, OrderedDict)):
        new_gradients = [new_gradients]

    for gradients in new_gradients:
        assert isinstance(gradients, (dict, OrderedDict))
        for key, val in gradients.items():
            if old_gradients.has_key(key):
                old_gradients[key] = old_gradients[key] + scale * val
            else:       
                old_gradients[key] = scale * val
    return old_gradients


def create_layers(layer_spec, data_dim, deterministic_layers=0, deterministic_act=None, deterministic_size=1.):
    """
    Parameters
    ----------
    layer_spec : str
        A specification for the layers to construct; typically takes a string
        like "100,50,25,10" and create P- and Q-models with  4 hidden layers
        of specified size.
    data_dim : int
        Dimensionality of the trainig/test data. The bottom-most layers
        will work with thgis dimension.
    deterministic_layers : int
        Dont want to talk about it.
    deterministic_act : 
    deterministic_size : float

    Returns
    -------
    p_layers : list
        List of ProbabilisticLayers with a ProbabilisticTopLayer on top.
    q_layers : list
        List of ProbabilisticLayers
    """
    inits = {
        'weights_init': RWSInitialization(factor=1.),
#        'weights_init': IsotropicGaussian(0.1),
        'biases_init': Constant(-1.0),
    }

    m = re.match("(\d*\.?\d*)x-(\d+)l-(\d+)", layer_spec)
    if m:
        first = int(data_dim * float(m.groups()[0]))
        last = float(m.groups()[2])
        n_layers = int(m.groups()[1])

        base = numpy.exp(numpy.log(first/last) / (n_layers-1))
        layer_sizes = [data_dim] + [int(last*base**i) for i in reversed(range(n_layers))]
        print(layer_sizes)
    else:
        layer_specs = [i for i in layer_spec.split(",")]
        layer_sizes = [data_dim] + [int(i) for i in layer_specs]

    p_layers = []
    q_layers = []
    for l, (size_lower, size_upper) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        """
        if size_upper < 0:
            lower_before_repeat = size_lower
            p = BernoulliLayer(
                    MLP([Sigmoid()], [size_lower, size_lower], **rinits), 
                    name="p_layer%d"%l)
            q = BernoulliLayer(
                    MLP([Sigmoid()], [size_lower, size_lower], **rinits), 
                    name="q_layer%d"%l)
            for r in xrange(-size_upper):
                p_layers.append(p)
                q_layers.append(q)
            continue
        elif size_lower < 0:
            size_lower = lower_before_repeat
        """
        size_mid = (deterministic_size * (size_upper + size_lower)) // 2

        p_layers.append(
            BernoulliLayer(
                MLP(
                    [deterministic_act() for i in range(deterministic_layers)]+[Logistic()],
                    [size_upper]+[size_mid for i in range(deterministic_layers)]+[size_lower],
                    **inits), 
                name="p_layer%d"%l))
        q_layers.append(
            BernoulliLayer(
                MLP(
                    [deterministic_act() for i in range(deterministic_layers)]+[Logistic()],
                    [size_lower]+[size_mid for i in range(deterministic_layers)]+[size_upper],
                    **inits), 
                name="q_layer%d"%l))

    p_layers.append(
        BernoulliTopLayer(
            layer_sizes[-1],
            name="p_top_layer",
            **inits))

    return p_layers, q_layers


#-----------------------------------------------------------------------------


def create_semi_layers(sup_layer_size , unsup_layer_size , data_dim, deterministic_size=0,y_dim = 10, deterministic_layers=0, deterministic_act=None ):
 
 
 
	inits = {
	    'weights_init': RWSInitialization(factor=1.),
    #        'weights_init': IsotropicGaussian(0.1),
	    'biases_init': Constant(-1.0),
	}

	#sup_layer_size = re.match("(\d*\.?\d*)x-(\d+)l-(\d+)", sup_layer_size)
	
	#unsup_layer_size = re.match("(\d*\.?\d*)x-(\d+)l-(\d+)", unsup_layer_size)
 
	sup_layer_size = [int(i) for i in sup_layer_size.split(",")]
	unsup_layer_size =[ data_dim]+ [int(i) for i in unsup_layer_size.split(",")]


      
# ----------------------------
        #self.p_y = MultinomialTop(...)
        p_y = MultinomialTopLayer_label(
            y_dim ,
            name="p_y",
            **inits) 
# ----------------------------
        p_h2_layers = []     # prior -> h2_dim           [BerrnoulliTop(...), Bernoulli(...)]
        for l, (size_lower, size_upper) in enumerate(zip(sup_layer_size[:len(sup_layer_size)], sup_layer_size[1:len(sup_layer_size)])):
	     
            size_mid = (deterministic_size * (size_upper + size_lower)) // 2

            p_h2_layers.append(
                BernoulliLayer(
                    MLP(
                         [Logistic()],#Activations
                        [size_upper]+[size_mid for i in range(deterministic_layers)]+[size_lower],#Dim
                        **inits), 
                    name="p_h2_layers%d"%l))
 
        p_h2_layers.append(
                BernoulliTopLayer(
                    sup_layer_size[-1],
                    name="p_h2_layers",
                    **inits))    
#--------------------------------
            
        #self.p_h1 = Bernoulli(...)   # y_dim+h2_dim -> h1_dim    Bernoulli(...)
        p_h1 =  BernoulliLayer(
                        MLP(
                             [Logistic()],#Activations
                            [y_dim + sup_layer_size[0]]+[size_mid for i in range(deterministic_layers)]+[unsup_layer_size[-1]],#Dim
                            **inits), 
                             name="p_h1"  )
#--------------------------------
        p_layers = [  ]        # h1_dim -> x_dim           [Bernoulli(...), ...]

        for l, (size_lower, size_upper) in enumerate(zip(unsup_layer_size[:len(unsup_layer_size)],  unsup_layer_size[1:len(unsup_layer_size)])):
            
            size_mid = (deterministic_size * (size_upper + size_lower)) // 2

            p_layers.append(
                BernoulliLayer(
                    MLP(
                        [Logistic()],#Activations
                        [size_upper]+[size_mid for i in range(deterministic_layers)]+[size_lower],#Dim
                        **inits), 
                    name="p_layers%d"%l))
 
#--------------------------------
        q_layers = []        # x_dim -> h1_dim          [Bernoulli(...)]
        for l, (size_lower, size_upper) in enumerate(zip(unsup_layer_size[:len(unsup_layer_size)],  unsup_layer_size[1:len(unsup_layer_size)])):
 
            size_mid = (deterministic_size * (size_upper + size_lower)) // 2
            q_layers.append(
                    BernoulliLayer(
                        MLP(
                            [Logistic()],
                            [size_lower]+[size_mid for i in range(deterministic_layers)]+[size_upper],
                            **inits), 
                        name="q_layer%d"%l))
#--------------------------------
        #self.q_y = Multinomial(...)  # h1_dim -> y_dim          Multinomial
        q_y =  MultinomialLayer(
	                    MLP(
                         [Logistic()],
                         [unsup_layer_size[-1]]+[size_mid for i in range(deterministic_layers)]+[y_dim],
                    		**inits), 
                          name="q_y " )      
        
#--------------------------------        
        q_h2_layers = []     # h1_dim+y_dim -> h2_dim   [Berrnoulli, ...]
   
        q_h2_layers.append(
                    BernoulliLayer(
                        MLP(
                             [Logistic()],
                            [unsup_layer_size[-1] + y_dim]+[size_mid for i in range(deterministic_layers)]+[sup_layer_size[0]],
                            **inits), 
                        name="q_h2_layers")) 
        for l, (size_lower, size_upper) in enumerate(zip(sup_layer_size[:len(sup_layer_size)], sup_layer_size[1:len(sup_layer_size)])):
 
            size_mid = (deterministic_size * (size_upper + size_lower)) // 2

            q_h2_layers.append(
                    BernoulliLayer(
                        MLP(
                            [Logistic()],
                            [size_lower]+[size_mid for i in range(deterministic_layers)]+[size_upper],
                            **inits), 
                        name="q_h2_layers%d"%l))
#--------------------------------  
        q_y_given_x =  MultinomialLayer(
	                    MLP(
                         [Logistic()],
                         [3]+[size_mid for i in range(deterministic_layers)]+[5],
                    		**inits), 
                          name="q_y " ) 
 
         

	return p_y, p_h1, p_h2_layers, p_layers, q_layers, q_y , q_h2_layers, q_y_given_x


#-----------------------------------------------------------------------------


class HelmholtzMachine(Initializable, Random):
    def __init__(self, p_layers, q_layers, **kwargs):
        super(HelmholtzMachine, self).__init__(**kwargs)
        
        self.p_layers = p_layers
        self.q_layers = q_layers

        self.children = p_layers + q_layers

    def _initialize(self):
        super(HelmholtzMachine, self)._initialize()

        """
        if not self.transpose_init:
            return

        p_layers = self.p_layers
        q_layers = self.q_layers
        
        for p, q in zip(p_layers[:-1], q_layers):
            if not hasattr(p, 'mlp'):
                continue
            if not hasattr(q, 'mlp'):
                continue
        
            p_trafos = p.mlp.linear_transformations
            q_trafos = q.mlp.linear_transformations

            logger.info("Transposed initialization for %s" % p)
            for ptrafo, qtrafo in zip(p_trafos, reversed(q_trafos)):
                Wp = ptrafo.W.get_value()
                Wq = qtrafo.W.get_value()

                assert Wp.shape == Wq.T.shape
                qtrafo.W.set_value(Wp.T)    
        """

#-----------------------------------------------------------------------------


class GradientMonitor(object):
    def __init__(self, gradients, prefix=""):
        self.gradients = gradients
        self.prefix = prefix
        pass

    def vars(self):
        prefix = self.prefix 
        monitor_vars = []

        aggregators = {
            'min':   tensor.min,
            'max':   tensor.max,
            'mean':  tensor.mean,
        }

        for key, value in six.iteritems(self.gradients):
            min = tensor.min(value)
            ipdb.set_trace()
            monitor_vars.append(monitor_vars)

        return monitor_vars
