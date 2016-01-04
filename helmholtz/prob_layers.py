
from __future__ import division, print_function 

import logging

import numpy
import theano

from collections import OrderedDict
from theano import tensor

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.roles import add_role, PARAMETER, WEIGHT, BIAS
from blocks.bricks import Random, MLP, Initializable, Softmax
from blocks.utils import pack, shared_floatx_zeros
from blocks.select import Selector

from distributions import bernoulli, Multinomial

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

N_STREAMS = 2048

 
#-----------------------------------------------------------------------------


class ProbabilisticTopLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticTopLayer, self).__init__(**kwargs)

    def sample_expected(self):
        raise NotImplemented

    def sample(self):
        raise NotImplemented

    def log_prob(self, X):
        raise NotImplemented

    def get_gradients(self, X, weights=1.):
        cost = -(weights * self.log_prob(X)).sum()
 
        params = Selector(self).get_parameters()
        
        gradients = OrderedDict()
        if isinstance(weights, float):
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X])
        else:
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, weights])
            
        return gradients
      
    def get_gradients_list(self,X_un,X, weights , alpha   , beta  , gamma , expectation_term=False):
        cost    = - (weights[1] * self.log_prob(X)).sum()
        cost_un = -alpha * 0.5 *  (weights[0] * self.log_prob(X_un)).sum()
        params = Selector(self).get_parameters()
        w = [X,X_un, weights[0], weights[1]]
        gradients = OrderedDict()
        
        if isinstance(weights, float):
            for pname, param in params.iteritems():
              if expectation_term:
                gradients[param] =   tensor.grad(cost, param, consider_constant=w) + tensor.grad(cost_un, param, consider_constant=w)-beta *   tensor.grad(self.log_prob(X_un, Y_un).mean(), param, consider_constant=w)- gamma* tensor.grad(self.log_prob(X, Y).mean(), param, consider_constant=w)
                
              else:
                gradients[param] =  tensor.grad(cost, param, consider_constant=w) + tensor.grad(cost_un, param, consider_constant=w)
                print("ProbabilisticTopLayer get_gradients_list part 1")
        else:
            for pname, param in params.iteritems():
              if expectation_term:
                cost_exp  = - gamma* tensor.grad(self.log_prob(X, Y).mean(), param, consider_constant=w)
                gradients[param] =   tensor.grad(cost, param, consider_constant=w)+   tensor.grad(cost_un, param, consider_constant=w)-beta *   tensor.grad(self.log_prob(X_un, Y_un).mean(), param, consider_constant=w)+cost_exp
                
              else:
                gradients[param] =    tensor.grad(cost, param, consider_constant=w)+   tensor.grad(cost_un, param, consider_constant=w)
        
        return gradients
      


class ProbabilisticLayer(Random):
    def __init__(self, **kwargs):
        super(ProbabilisticLayer, self).__init__(**kwargs)

    def sample_expected(self, Y):
        raise NotImplemented

    def sample(self, Y):
        raise NotImplemented

    def log_prob(self, X, Y):
        raise NotImplemented

    def get_gradients(self, X, Y, weights=1.):
        cost = -(weights * self.log_prob(X, Y)).sum()
        
        params = Selector(self).get_parameters()

        gradients = OrderedDict()
        if isinstance(weights, float):
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, Y])
        else:
            for pname, param in params.iteritems():
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, Y, weights])
            
        return gradients
      
    def get_gradients_list(self,X_un, Y_un, X, Y, weights , alpha  , beta  ,gamma,expectation_term =False):
        cost    = - (weights[1] * self.log_prob(X, Y)).sum()
        cost_un = - alpha * 0.5 * (weights[0] * self.log_prob(X_un, Y_un)).sum()
        
        w = [X_un, Y_un, X, Y, weights[0], weights[1]]
          
        params = Selector(self).get_parameters()
        
        gradients = OrderedDict()
        
        if isinstance(weights, float):
          
            for pname, param in params.iteritems():
              if expectation_term:
                cost_exp =- gamma * tensor.grad(self.log_prob(X, Y).mean(), param, consider_constant=w)
                gradients[param] = tensor.grad(cost, param, consider_constant=[X, Y]) + tensor.grad(cost_un, param, consider_constant=w)- beta * tensor.grad(self.log_prob(X_un, Y_un).mean(), param, consider_constant=w)  + cost_exp
                
              else:
                gradients[param] = tensor.grad(cost, param, consider_constant=w) + tensor.grad(cost_un, param, consider_constant=w)
                print("get_gradients_list part 1")
           
        else:
            for pname, param in params.iteritems():
              if expectation_term:      
                cost_exp =- gamma * tensor.grad(self.log_prob(X, Y).mean(), param, consider_constant=w)
                gradients[param] = tensor.grad(cost, param, consider_constant=w) + tensor.grad(cost_un, param, consider_constant=w) - beta * tensor.grad(self.log_prob(X_un, Y_un).mean(), param, consider_constant=w)+ cost_exp
                
              else: 
                gradients[param] = tensor.grad(cost, param, consider_constant=w) + tensor.grad(cost_un, param, consider_constant=w)
                 
                
        return gradients
      
    def get_gradients_list_h1_type3(self,samples_y_unsup, samples_y_sup, h1, weights , alpha  , beta   ,gamma ,expectation_term =False):
        w_unsup = weights[0]
        w_sup   = weights[1]     
        cost_sup   = -(w_sup * self.log_prob(samples_y_sup, h1)).sum()
        cost_unsup = -  alpha * 0.5 * (w_unsup * self.log_prob(samples_y_unsup, h1)).sum()
           
        params = Selector(self).get_parameters()
        
        gradients = OrderedDict()
        w = [samples_y_sup, h1,samples_y_unsup,w_unsup,w_sup]
        if isinstance(weights, float):
          
            for pname, param in params.iteritems():
              if expectation_term:
                grad_exp_sup =-  gamma* tensor.grad(self.log_prob(samples_y_sup, h1).mean(), param, consider_constant=w)
                grad_exp_unsup =- beta * tensor.grad(self.log_prob(samples_y_unsup, h1).mean(), param, consider_constant=w)
                
                grad_sup =   tensor.grad(cost_sup, param, consider_constant=w)
                grad_unsup =     tensor.grad(cost_unsup, param, consider_constant=w)
                
                gradients[param] = grad_unsup + grad_sup + grad_exp_unsup + grad_exp_sup
                
              else:
                print("get_gradients_list_h1_type3 else part 1")
                grad_sup =  tensor.grad(cost_sup, param, consider_constant=w)
                grad_unsup =    tensor.grad(cost_unsup, param, consider_constant=w)
                
                gradients[param] = grad_unsup + grad_sup 
        else:
            for pname, param in params.iteritems():
                if expectation_term:
                  grad_exp_sup =-  gamma * tensor.grad(self.log_prob(samples_y_sup, h1).mean(), param, consider_constant=w)
                  grad_exp_unsup =-  beta * tensor.grad(self.log_prob(samples_y_unsup, h1).mean(), param, consider_constant=w)
                  
                  grad_sup =  tensor.grad(cost_sup, param, consider_constant=w)
                  grad_unsup =   tensor.grad(cost_unsup, param, consider_constant=w)
                  
                  gradients[param] = grad_unsup + grad_sup + grad_exp_unsup + grad_exp_sup
                  
                else:
                  print("get_gradients_list_h1_type3 else part 2")
                  grad_sup =   tensor.grad(cost_sup, param, consider_constant=w)
                  grad_unsup =    tensor.grad(cost_unsup, param, consider_constant=w)
                  
                  gradients[param] = grad_unsup + grad_sup 
                 
        return gradients

    def get_gradients_list_h1(self, h_p, h_n, weights , alpha  , beta   ,gamma  ):
        w_unsup = weights[0]
        w_sup = weights[1]
       
 
        cost_sup = -(w_sup * self.log_prob( h_p, h_n)).sum()
        cost_unsup = -alpha*0.5 * (w_unsup * self.log_prob( h_p, h_n)).sum()
          
        params = Selector(self).get_parameters()
        
        gradients = OrderedDict()
        w = [ h_p, h_n,w_unsup,w_sup]
        if isinstance(weights, float):
          
            for pname, param in params.iteritems():
 
                
                grad_sup =    tensor.grad(cost_sup, param, consider_constant=w)
                grad_unsup =     tensor.grad(cost_unsup, param, consider_constant=w)
                
                gradients[param] = grad_unsup + grad_sup  
        else:
          
           
            for pname, param in params.iteritems():
 
                
                grad_sup =    tensor.grad(cost_sup, param, consider_constant=w)
                grad_unsup =      tensor.grad(cost_unsup, param, consider_constant=w)
                
                gradients[param] = grad_unsup + grad_sup  
                 
        return gradients
      
      

#-----------------------------------------------------------------------------


class BernoulliTopLayer(Initializable, ProbabilisticTopLayer):
    def __init__(self, dim_X, biases_init, **kwargs):
        super(BernoulliTopLayer, self).__init__(**kwargs)
        self.sigmoid_frindge = 1e-6
        self.dim_X = dim_X
        self.biases_init = biases_init

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        self.add_auxiliary_variable(b.norm(2), name='b_norm')
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=[], outputs=['X_expected'])
    def sample_expected(self):
        b = self.parameters[0]
        return tensor.nnet.sigmoid(b) #.clip(self.sigmoid_frindge, 1.-self.sigmoid_frindge)

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        prob_X = self.sample_expected()
        prob_X = tensor.zeros((n_samples, prob_X.shape[0])) + prob_X
        X = bernoulli(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        prob_X = self.sample_expected()
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1.-prob_X)
        return log_prob.sum(axis=1)


class BernoulliLayer(Initializable, ProbabilisticLayer):
    def __init__(self, mlp, **kwargs):
        super(BernoulliLayer, self).__init__(**kwargs)

        self.mlp = mlp

        self.dim_X = mlp.output_dim
        self.dim_Y = mlp.input_dim
        self.sigmoid_frindge = 1e-6

        self.children = [self.mlp]

    @application(inputs=['Y'], outputs=['X_expected'])
    def sample_expected(self, Y):
        return self.mlp.apply(Y) #.clip(self.sigmoid_frindge, 1.-self.sigmoid_frindge)

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        prob_X = self.sample_expected(Y)
        X = bernoulli(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X, Y)
      
    def sample_concatenate(self, h2, y):    
	H_concatenate = tensor.concatenate([h2, y], axis =1)
	 
        prob_X = self.sample_expected(H_concatenate)
        X = bernoulli(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X, H_concatenate)


    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        prob_X = self.sample_expected(Y)
        log_prob = X*tensor.log(prob_X) + (1.-X)*tensor.log(1-prob_X)
        return log_prob.sum(axis=1)

#-----------------------------------------------------------------------------


class GaussianTopLayer(Initializable, ProbabilisticTopLayer):
    def __init__(self, dim_X, fixed_sigma=None, **kwargs):
        super(GaussianTopLayer, self).__init__(**kwargs)
        self.fixed_sigma = fixed_sigma
        self.dim_X = dim_X

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIAS)
        self.parameters = [b]
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=[], outputs=['mean', 'log_sigma'])
    def sample_expected(self, n_samples):
        b, = self.parameters
        mean      = tensor.zeros((n_samples, self.dim_X))
        #log_sigma = tensor.zeros((n_samples, self.dim_X)) + b
        log_sigma = tensor.log(self.fixed_sigma)

        return mean, log_sigma

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        mean, log_sigma = self.sample_expected(n_samples)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=(n_samples, self.dim_X),
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X)

    @application(inputs='X', outputs='log_prob')
    def log_prob(self, X):
        mean, log_sigma = self.sample_expected(X.shape[0])

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)


#-----------------------------------------------------------------------------


class GaussianLayerFixedSigma(Initializable, ProbabilisticLayer):
    def __init__(self, dim_X, mlp, sigma=None, **kwargs):
        super(GaussianLayerFixedSigma, self).__init__(**kwargs)
        self.mlp = mlp
        self.dim_X = dim_X
        self.dim_Y = mlp.input_dim
        self.dim_H = mlp.output_dim
        self.sigma = sigma

        self.children = [self.mlp]
        

    def _allocate(self):
        super(GaussianLayerFixedSigma, self)._allocate()

        dim_X, dim_H = self.dim_X, self.dim_H

        self.W_mean = shared_floatx_zeros((dim_H, dim_X), name='W_mean')
        add_role(self.W_mean, WEIGHT)

        self.b_mean = shared_floatx_zeros((dim_X,), name='b_mean')
        add_role(self.b_mean, BIAS)

        self.parameters = [self.W_mean, self.b_mean]
        
    def _initialize(self):
        super(GaussianLayerFixedSigma, self)._initialize()

        W_mean, b_mean = self.parameters

        self.weights_init.initialize(W_mean, self.rng)
        self.biases_init.initialize(b_mean, self.rng)

    @application(inputs=['Y'], outputs=['mean', 'log_sigma'])
    def sample_expected(self, Y):
        W_mean, b_mean = self.parameters

        a = self.mlp.apply(Y)
        mean      = tensor.dot(a, W_mean) + b_mean
        log_sigma = tensor.log(self.sigma)

        return mean, log_sigma

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)


#-----------------------------------------------------------------------------


class GaussianLayer(Initializable, ProbabilisticLayer):
    def __init__(self, dim_X, mlp, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.mlp = mlp
        self.dim_X = dim_X
        self.dim_Y = mlp.input_dim
        self.dim_H = mlp.output_dim

        self.children = [self.mlp]

    def _allocate(self):
        super(GaussianLayer, self)._allocate()

        dim_X, dim_Y, dim_H = self.dim_X, self.dim_Y, self.dim_H

        self.W_mean = shared_floatx_zeros((dim_H, dim_X), name='W_mean')
        self.W_ls   = shared_floatx_zeros((dim_H, dim_X), name='W_ls')
        add_role(self.W_mean, WEIGHT)
        add_role(self.W_ls, WEIGHT)

        self.b_mean = shared_floatx_zeros((dim_X,), name='b_mean')
        self.b_ls   = shared_floatx_zeros((dim_X,), name='b_ls')
        add_role(self.b_mean, BIAS)
        add_role(self.b_ls, BIAS)

        self.parameters = [self.W_mean, self.W_ls, self.b_mean, self.b_ls]
        
    def _initialize(self):
        super(GaussianLayer, self)._initialize()

        W_mean, W_ls, b_mean, b_ls = self.parameters

        self.weights_init.initialize(W_mean, self.rng)
        self.weights_init.initialize(W_ls, self.rng)
        self.biases_init.initialize(b_mean, self.rng)
        self.biases_init.initialize(b_ls, self.rng)

    @application(inputs=['Y'], outputs=['mean', 'log_sigma'])
    def sample_expected(self, Y):
        W_mean, W_ls, b_mean, b_ls = self.parameters

        a = self.mlp.apply(Y)
        mean = tensor.dot(a, W_mean) + b_mean
        log_sigma = tensor.dot(a, W_ls) + b_ls

        return mean, log_sigma

    @application(inputs=['Y'], outputs=['X', 'log_prob'])
    def sample(self, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Sample from mean-zeros std.-one Gaussian
        U = self.theano_rng.normal(
                    size=mean.shape, 
                    avg=0., std=1.)
        # ... and scale/translate samples
        X = mean + tensor.exp(log_sigma) * U

        return X, self.log_prob(X, Y)

    @application(inputs=['X', 'Y'], outputs=['log_prob'])
    def log_prob(self, X, Y):
        mean, log_sigma = self.sample_expected(Y)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*tensor.log(2*numpy.pi) - log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)

        return log_prob.sum(axis=1)

    def get_gradients(self, X, Y, weights=1.):
        W_mean, W_ls, b_mean, b_ls = self.parameters

        mean, log_sigma = self.sample_expected(Y)
        sigma = tensor.exp(log_sigma)

        cost = -log_sigma -0.5*(X-mean)**2 / tensor.exp(2*log_sigma)
        if weights != 1.:
            cost = -weights.dimshuffle(0, 'x') * cost

        cost_scaled = sigma**2 * cost
        cost_gscale = (sigma**2).sum(axis=1).dimshuffle([0, 'x'])   
        cost_gscale = cost_gscale * cost
        
        gradients = OrderedDict()

        params = Selector(self.mlp).get_parameters()
        for pname, param in params.iteritems():
            gradients[param] = tensor.grad(cost_gscale.sum(), param, consider_constant=[X, Y])

        gradients[W_mean] = tensor.grad(cost_scaled.sum(), W_mean, consider_constant=[X, Y])
        gradients[b_mean] = tensor.grad(cost_scaled.sum(), b_mean, consider_constant=[X, Y])

        gradients[W_ls] = tensor.grad(cost_scaled.sum(), W_ls, consider_constant=[X, Y])
        gradients[b_ls] = tensor.grad(cost_scaled.sum(), b_ls, consider_constant=[X, Y])
            
        return gradients
      
      
#------------------------------------------------------------------------------------
 
 
class MultinomialTopLayer(Initializable, ProbabilisticTopLayer):
    def __init__(self, dim_X, biases_init, **kwargs):
        super(MultinomialTopLayer, self).__init__(**kwargs)
        self.dim_X = dim_X
        self.biases_init = biases_init

    def _allocate(self):
        b = shared_floatx_zeros((self.dim_X,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)
        self.add_auxiliary_variable(b.norm(2), name='b_norm')
        
    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=['X'], outputs=['log_prob'])
    def log_prob(self, X): #x is the labels
        b = self.parameters[0]
        prob_X = tensor.nnet.softmax(b)
        log_prob = X*tensor.log(prob_X)  
        return log_prob.sum(axis=1)

    def sample_expected(self, n_samples):
        b = self.parameters[0]
        prob_X = tensor.nnet.softmax(b)
        prob_X = tensor.zeros((n_samples, prob_X.shape[0])) + prob_X 
        return prob_X

    @application(outputs=['X', 'log_prob'])
    def sample(self, n_samples):
        prob_X = self.sample_expected(n_samples)
        X = Multinomial(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X, self.log_prob(X)
       
 
#------------------------------------------------------------------------------------


class MultinomialLayer(Initializable, ProbabilisticLayer):
    def __init__(self, mlp, **kwargs):
        super(MultinomialLayer, self).__init__(**kwargs)

        self.mlp = mlp
        self.dim_X = mlp.output_dim
        self.dim_Y = mlp.input_dim
        self.children = [self.mlp]
 
    def sample_expected(self, Y):
        return tensor.nnet.softmax(self.mlp.apply(Y))
 
    def sample(self, Y):
        prob_X = self.sample_expected(Y)
        X = Multinomial('auto')(prob_X, rng=self.theano_rng, nstreams=N_STREAMS)
        return X , self.log_prob(X, Y)

    def log_prob(self, X, Y): #x is the labels
        prob_X = self.sample_expected(Y)
        log_prob = X*tensor.log(prob_X) 
        return log_prob.sum(axis=1)
