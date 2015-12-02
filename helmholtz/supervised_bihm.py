
from __future__ import division, print_function 

import sys
sys.path.append("../")

import re
import logging

import numpy
import theano

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
from helmholtz import HelmholtzMachine
from helmholtz import merge_gradients, replicate_batch, logsumexp, flatten_values, unflatten_values 
from prob_layers import BernoulliTopLayer, BernoulliLayer, MultinomialTopLayer_label, MultinomialLayer
logger = logging.getLogger(__name__)
floatX = theano.config.floatX
import numpy as np

#-----------------------------------------------------------------------------


class  Supervised_BiHM(HelmholtzMachine):
    def __init__(self, p_y, p_h1, p_layers, q_layers, q_y,   q_y_given_x, x_dim, y_dim ,sup_layer_size, unsup_layer_size, **kwargs):
        super( Supervised_BiHM, self).__init__(p_layers, q_layers, **kwargs)

        self.p_y = p_y 
        self.p_h1 = p_h1
 
        self.p_layers = p_layers
        self.q_layers = q_layers
        self.q_y = q_y 
 
        self.q_y_given_x = q_y_given_x
  
        self.children = [p_y] + [p_h1] +   p_layers + q_layers + [q_y] + [q_y_given_x]

        self.y_dim = y_dim
        self.sup_layer_size   =  sup_layer_size
        self.unsup_layer_size = unsup_layer_size
        self.p_y = p_y 
        self.p_h1 = p_h1
 
        self.p_layers = p_layers
        self.q_layers = q_layers
        self.q_y = q_y 
 
        self.q_y_given_x = q_y_given_x
        
      
    def log_prob_p(self, samples, h2, label):  
        n_unsup_layers =  len(self.q_layers)
        n_sup_layers =  len(self.p_h2_layers) 
        n_samples = samples[0].shape[0]

        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
        
        log_p_h1 = [] #p(h1| h2,y)
 
        
        loh_p_y = self.p_y.log_prob( label)
        samples_sup = tensor.concatenate([h2[0], label])
        log_p_h1 = self.p_h1.log_prob(samples[n_unsup_layers-1], samples_sup)
 
        for l in xrange(n_unsup_layers-1 ):

            log_p_layers[l] = self.p_layers[l].log_prob(samples[l], samples[l+1])
 
        for l in xrange(n_sup_layers-1):
            #print(l)
            log_p_h2_layers [l] = self.p_h2_layers[l].log_prob(h2[l], h2[l+1])
                
        log_p_h2_layers [n_sup_layers-1] = self.p_h2_layers [n_sup_layers-1].log_prob(h2[n_sup_layers-1])

        return log_p_layers, log_p_y ,log_p_h1 , log_p_h2_layers
        
    def log_prob_q(self, samples, h2, label):
 
        n_unsup_layers = len(self.q_layers)
        n_sup_layers = len(self.p_h2_layers) 
        
        n_samples = samples[0].shape[0]
      
        
        log_q_layers = [None] * n_unsup_layers
        log_q_h2_layers = [ None]*n_sup_layers
        log_q_y = []
        
        log_q_y = self.q_y.log_prob(label, samples[n_unsup_layers-1])
        
        log_q[0] = tensor.zeros([n_samples])
        for l in xrange(n_unsup_layers-1):
            log_q_layers[l+1] = self.q_layers[l].log_prob(samples[l+1], samples[l])
        
        log_q_h2_layers[0] = tensor.zeros([n_samples])
        for l in xrange(n_sup_layers-1):
            log_q_h2_layers[l+1] = self.q_h2_layers[l].log_prob(h2[l+1], h2[l])

        return log_q_layers, log_q_y, log_q_h2_layers
  
    def sample_p_unsup(self, n_samples):
 
        n_unsup_layers = len(self.q_layers)
        n_sup_layers =  len(self.p_h2_layers) 
     

        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
        p_h2_layers = self.p_h2_layers
        
        q_layers = self.q_layers
        q_h2_layers = self.q_h2_layers
        q_y = self.q_y 
        
      
        samples_y =[]
        samples_layers =[None]* n_unsup_layers
        samples_h2 = [None] * n_sup_layers
 
        
        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
 
        log_p_h2_layers =[None] * n_sup_layers
 
        samples_y, log_p_y = p_y.sample(n_samples)
        samples_layers [n_unsup_layers-1], log_p_layers[n_unsup_layers-1] = p_h1.sample_concatenate(n_samples, samples_h2[0],samples_y )
        
         
        for l in reversed(xrange(1, n_unsup_layers)):
            samples_layers[l-1], log_p[l-1] = p_layers[l-1].sample(samples_layers[l]) 

        samples_h2 [n_sup_layers-1], log_p_h2_layers[n_sup_layers-1] = p_h2_layers[n_sup_layers-1].sample(n_samples)        
        for l in reversed(xrange(1, n_sup_layers)):
            samples_h2[l-1],log_p_h2_layers[l-1] = p_h2_layers[l-1].sample(samples_h2[l])   

 
        log_q_layers, log_q_y, log_q_h2_layers = self.log_prob_q(samples_layers,samples_h2,samples_y ) #log_prob_q(self, samples, h2, label)
    
        return samples_y,samples_layers, samples_h2,log_p_layers,log_p_y, log_p_h2_layers, log_q_layers, log_q_y, log_q_h2_layers
    def sample_p_sup(self, n_samples, label):
  
        n_unsup_layers =  len(self.q_layers)
        n_sup_layers =  len(self.p_h2_layers) 
     

        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
        p_h2_layers = self.p_h2_layers
        
        q_layers = self.q_layers
        q_h2_layers = self.q_h2_layers
        q_y = self.q_y 
        
      
        samples_y = replicate_batch(label, n_samples) 
        samples_layers =[None]* n_unsup_layers
        samples_h2 = [None] * n_sup_layers
   
        
        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
 
        log_p_h2_layers =[None] * n_sup_layers
 
        log_q_y = q_y.log_prob(samples_y)
        samples_layers [n_unsup_layers-1], log_p_layers[n_unsup_layers-1] = p_h1.sample_concatenate(n_samples, samples_h2[0],samples_y )
        
          
        for l in reversed(xrange(1, n_unsup_layers)):
            samples_layers[l-1], log_p[l-1] = p_layers[l-1].sample(samples_layers[l]) 

        samples_h2 [n_sup_layers-1], log_p_h2_layers[n_sup_layers-1] = p_h2_layers[n_sup_layers-1].sample(n_samples)        
        for l in reversed(xrange(1, n_sup_layers)):
            samples_h2[l-1],log_p_h2_layers[l-1] = p_h2_layers[l-1].sample(samples_h2[l])   

 
        log_q_layers, log_q_y, log_q_h2_layers = self.log_prob_q(samples_layers,samples_h2,samples_y )  
        
        
        return samples_y,samples_layers, samples_h2,log_p_layers,log_p_y, log_p_h2_layers, log_q_layers, log_q_y, log_q_h2_layers
 

    def sample_q_unsup(self, features, n_samples, samples_h1, label): #Todo
        n_unsup_layers = len(self.q_layers)
   
     
        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
   
        
        q_layers = self.q_layers
   
        q_y = self.q_y 
        
      
        samples_y =  [] # 
        samples_layers = samples_h1 
 
 
        
        log_q_layers = [None] * (n_unsup_layers+1)#p(h1_i| h1_i+1)
        log_q_y = []
 
 
        batch_size = features.shape[0]
 
 
        log_q_layers[0] = tensor.zeros([batch_size])
        for l in xrange(n_unsup_layers):
            _, log_q_layers[l+1] = q_layers[l].sample(samples_layers[l])
               
 
 
        samples_y, log_q_y = q_y.sample(samples_layers[n_unsup_layers])
       # _, log_q_y = q_y.sample(samples_layers[n_unsup_layers]) #Todo
 
        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
        log_p_h1 = [] #p(h1| h2,y)
 
        
        log_p_y  = p_y.log_prob(samples_y )
 
        for l in reversed(range(1, n_unsup_layers)):
            log_p_layers[l-1] = p_layers[l-1].log_prob(samples_layers[l-1], samples_layers[l]) 
        log_p_layers[n_unsup_layers-1] = p_layers[n_unsup_layers-1].log_prob(samples_layers[n_unsup_layers-1], samples_layers[n_unsup_layers]) 
 
        log_p_h1 = self.p_h1.log_prob(samples_layers[n_unsup_layers], samples_y)
  
        return  samples_y, log_p_layers, log_p_y ,log_p_h1 ,  log_q_layers, log_q_y 
        

    def sample_q_sup (self, features, label ):
        n_unsup_layers =  len(self.q_layers) 
 
         
        p_layers = self.p_layers #p(h1_i| h1_i+1)

        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
 
        
        q_layers = self.q_layers
 
        q_y = self.q_y 
        
 
        samples_y  =  label# replicate_batch(label, n_samples) 
     
        samples_layers =[None]* (n_unsup_layers+1)
 
       
        
        log_q_layers = [None] * (n_unsup_layers+1) #p(h1_i| h1_i+1)
        log_q_y = []
   
 
        
        batch_size = features.shape[0]

        samples_layers[0] = features 
        log_q_layers[0] = tensor.zeros([batch_size])

        for l in xrange(n_unsup_layers):
            samples_layers[l+1], log_q_layers[l+1] = q_layers[l].sample(samples_layers[l])

 
        log_q_y = q_y.log_prob( samples_y, samples_layers[n_unsup_layers])

  
        log_p_layers = [None] *  n_unsup_layers  #p(h1_i| h1_i+1)
        log_p_y = []
        log_p_h1 = [] #p(h1| h2,y)
 
 
        log_p_y  = p_y.log_prob(samples_y )
 
 
        for l in reversed(range(1, n_unsup_layers)):
            log_p_layers[l-1] = p_layers[l-1].log_prob(samples_layers[l-1], samples_layers[l]) 
         
        log_p_layers[n_unsup_layers-1] = p_layers[n_unsup_layers-1].log_prob(samples_layers[n_unsup_layers-1], samples_layers[n_unsup_layers]) 
 
        log_p_h1 = self.p_h1.log_prob(samples_layers[n_unsup_layers],samples_y)
      
        return samples_layers, samples_y,  log_p_layers, log_p_y ,log_p_h1 ,   log_q_layers, log_q_y


    def sample_h1_y (self, features, n_samples, h2):
        n_unsup_layers =  len(self.q_layers)
        n_sup_layers =  len(self.p_h2_layers) 

        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
        p_h2_layers = self.p_h2_layers
        
        q_layers = self.q_layers
        q_h2_layers = self.q_h2_layers
        q_y = self.q_y 
        
      
        samples_y =   []
        samples_layers =[None]* (n_unsup_layers+1)
        samples_h2 = [None] * n_sup_layers
        #samples_h1 = []
        
        log_q_layers = [None] *    (n_unsup_layers+1) #p(h1_i| h1_i+1)
        log_q_y = []
        #log_p_h1 = [] #p(h1| h2,y)
        log_q_h2_layers =[None] * n_sup_layers
 
        batch_size = features.shape[0]

        samples_layers[0] = features 
        log_q_layers[0] = tensor.zeros([batch_size])
 
        for l in xrange(n_unsup_layers):
            samples_layers[l+1], log_q_layers[l+1] = q_layers[l].sample(samples_layers[l])
            
        samples_y, log_q_y = q_y.sample(samples_layers[n_unsup_layers]) 
            
        samples_h2_layers = h2#[0],             
         
        
        _, log_q_h2_layers[0] = q_h2_layers[0].sample_concatenate(samples_layers[n_unsup_layers],samples_y)
    

        for l in xrange(n_sup_layers-1):
            _, log_q_h2_layers[l+1] = q_h2_layers[l+1].sample(samples_h2_layers[l])
 
        
        # get log-probs from generative model
        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
        log_p_h1 = [] #p(h1| h2,y)
        log_p_h2_layers =[None] * n_sup_layers
        
        log_p_y  = p_y.log_prob(samples_y )
        log_p_h2_layers[n_sup_layers-1] = p_h2_layers[n_sup_layers-1].log_prob(samples_h2_layers[n_sup_layers-1])
        for l in reversed(range(1, n_sup_layers)):
            log_p_h2_layers[l-1] = p_h2_layers[l-1].log_prob(samples_h2_layers[l-1], samples_h2_layers[l])
        
        for l in reversed(range(1, n_unsup_layers)):
            log_p_layers[l-1] = p_layers[l-1].log_prob(samples_layers[l-1], samples_layers[l]) 

        log_p_layers[n_unsup_layers-1] = p_layers[n_unsup_layers-1].log_prob(samples_layers[n_unsup_layers-1], samples_layers[n_unsup_layers]) 
        samples_sup = tensor.concatenate([samples_h2_layers[0], samples_y ], axis=1)
        log_p_h1 = self.p_h1.log_prob(samples_layers[n_unsup_layers], samples_sup)   
       
        return samples_layers, samples_y,   log_p_layers, log_p_y ,log_p_h1 , log_p_h2_layers, log_q_layers, log_q_y, log_q_h2_layers


    def sample_y (self, features, n_samples, h1):
        n_unsup_layers =  len(self.q_layers)
 
     

        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
 
        
        q_layers = self.q_layers
 
        q_y = self.q_y 
        
      
        samples_y =   []
        samples_layers =[None]* (n_unsup_layers+1)
 
        #samples_h1 = []
        
        log_q_layers = [None] *    (n_unsup_layers+1) #p(h1_i| h1_i+1)
        log_q_y = []
 
 
        batch_size = features.shape[0]
 
        samples_layers = h1 
        log_q_layers[0] = tensor.zeros([batch_size])
 
        for l in xrange(n_unsup_layers):
            _, log_q_layers[l+1] = q_layers[l].sample(samples_layers[l])
            
        samples_y, log_q_y = q_y.sample(samples_layers[n_unsup_layers]) 
 
 
        log_p_layers = [None] * n_unsup_layers #p(h1_i| h1_i+1)
        log_p_y = []
        log_p_h1 = [] #p(h1| h2,y)
 

        log_p_y  = p_y.log_prob(samples_y )
 
  
        for l in reversed(range(1, n_unsup_layers)):
            log_p_layers[l-1] = p_layers[l-1].log_prob(samples_layers[l-1], samples_layers[l]) 

        log_p_layers[n_unsup_layers-1] = p_layers[n_unsup_layers-1].log_prob(samples_layers[n_unsup_layers-1], samples_layers[n_unsup_layers]) 
 
        log_p_h1 = self.p_h1.log_prob(samples_layers[n_unsup_layers], samples_y)   
       
        return   samples_y,   log_p_layers, log_p_y ,log_p_h1 ,   log_q_layers, log_q_y 

        
    def sup_lower_bound(self, features, label, n_samples):
        """It returns p_tild_star(x,y) of size ( batch_size ) """
        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
 
        
        q_layers = self.q_layers
 
        q_y = self.q_y 
        q_y_given_x = self.q_y_given_x
        
        n_unsup_layers = len(self.q_layers)
 
  
        
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)
        label =  replicate_batch(label, n_samples)
        samples_layers, samples_y,  log_p_layers, log_p_y ,log_p_h1 ,  log_q_layers, log_q_y  = self.sample_q_sup (x, label )
 
       
        log_p_layers     = unflatten_values(log_p_layers, batch_size, n_samples)
 
        log_p_y          = unflatten_values([log_p_y], batch_size, n_samples)
        log_p_h1         = unflatten_values([log_p_h1], batch_size, n_samples)
 
        log_q_y          = unflatten_values([log_q_y], batch_size, n_samples)     #( batch_size, n_samples)
        log_q_layers     = unflatten_values(log_q_layers, batch_size, n_samples)  #( n_unsup_layers, batch_size, n_samples)
 
         
        prob_layes = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers:
            prob_layes  =  p + prob_layes 
        
 
            
        prob_q_layes = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers:
            prob_q_layes=  p +   prob_q_layes


 
   
   
        all_prob = 1./2*(prob_layes +   log_p_y + log_p_h1  +log_q_y - prob_q_layes   )
        sup_lower_bound =  logsumexp( all_prob, axis=-1 ) -tensor.log(n_samples ) 
 
        return  samples_layers, sup_lower_bound  
        

    def unsup_lower_bound(self,features,  n_samples, samples_h1): 
        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)

        
        q_layers = self.q_layers

        q_y = self.q_y 
        
        n_unsup_layers =  len(self.q_layers)

  
        
        batch_size = features.shape[0]

        x = replicate_batch(features, n_samples)
 
        samples_y, log_p_layers, log_p_y, log_p_h1, log_q_layers, log_q_y  = self.sample_y(x,  n_samples, samples_h1)
 
        log_p_layers     = unflatten_values(log_p_layers, batch_size, n_samples)
        log_p_y          = unflatten_values([log_p_y], batch_size, n_samples)
        log_p_h1         = unflatten_values([log_p_h1], batch_size, n_samples)
        
        log_q_y          = unflatten_values([log_q_y], batch_size, n_samples)
        log_q_layers     = unflatten_values(log_q_layers, batch_size, n_samples)
 
 
       
        prob_layes = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers:
            prob_layes  =  p + prob_layes 
        

            
        prob_q_layes = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers:
            prob_q_layes=  p +   prob_q_layes


   
   
        all_prob = 1./2*(prob_layes + log_p_y + log_p_h1  -log_q_y - prob_q_layes  )
        unsup_lower_bound =2.0* logsumexp( all_prob, axis=-1 ) -2.0*tensor.log(n_samples )
 
        return unsup_lower_bound  
 

    def log_likelihood(self, features, label, n_samples, mask):
        p_layers = self.p_layers #p(h1_i| h1_i+1)
        p_y = self.p_y
        p_h1 = self.p_h1 #p(h1| h2,y)
 
        
        q_layers = self.q_layers
 
        q_y = self.q_y 
        
        n_unsup_layers =  len(self.q_layers)
       
 
        
        batch_size = features.shape[0]
 
        samples_layers, sup_lower_bound  =  self.sup_lower_bound(features, label, n_samples) 
        sup_lower_bound = sup_lower_bound.reshape((batch_size,1)) 
        
        unsup_lower_bound  = self.unsup_lower_bound(features,   n_samples, samples_layers ) 
        unsup_lower_bound = unsup_lower_bound.reshape((batch_size,1))
       
        sup_lower_bound  = 0.5*unsup_lower_bound + sup_lower_bound
     
        lower_bound = tensor.switch(mask, sup_lower_bound, unsup_lower_bound) #- alpha*tensor.switch(mask, sup_log_q_y_given_x, unsup_log_q_y_given_x)
 
        return  lower_bound 
        

    def sup_importance_weights(self,     log_p_layers,log_p_y ,log_p_h1 ,  log_q_layers, log_q_y,  batch_size, n_samples):
   
        log_p_layers     = unflatten_values(log_p_layers, batch_size, n_samples)
        log_p_y          = unflatten_values([log_p_y], batch_size, n_samples)
        log_p_h1         = unflatten_values([log_p_h1], batch_size, n_samples)
        log_q_y          = unflatten_values([log_q_y], batch_size, n_samples)
        log_q_layers     = unflatten_values(log_q_layers, batch_size, n_samples)
  
        prob_layes = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers:
            prob_layes  =  p + prob_layes 
        
 
            
        prob_q_layes = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers:
            prob_q_layes=  p +   prob_q_layes

   
        prob_all  = 0.5*(prob_layes +  log_p_y + log_p_h1 +log_q_y - prob_q_layes  ) -tensor.log(n_samples )
        
        log_w = prob_all - tensor.shape_padright(logsumexp( prob_all, axis =-1))   
  
        w = 0.5*tensor.exp(log_w)
        
        return w 


    def unsup_importance_weights(self,   log_p_layers,log_p_y ,log_p_h1 ,  log_q_layers, log_q_y, batch_size, n_samples):
 
        log_p_layers     = unflatten_values(log_p_layers, batch_size, n_samples)
        log_p_y          = unflatten_values([log_p_y], batch_size, n_samples)
        log_p_h1         = unflatten_values([log_p_h1], batch_size, n_samples)
 
        log_q_y          = unflatten_values([log_q_y], batch_size, n_samples)
        log_q_layers     = unflatten_values(log_q_layers, batch_size, n_samples)
    
 
       
        prob_layes = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers:
            prob_layes  =  p + prob_layes 
        
 
        prob_q_layes = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers:
            prob_q_layes=  p +   prob_q_layes

 
   
   
        prob_all  = 0.5*(prob_layes  + log_p_y + log_p_h1 -log_q_y - prob_q_layes    )  -tensor.log(n_samples )
 
        
        log_w = prob_all - tensor.shape_padright(logsumexp( prob_all, axis =-1))#plus
  
        w = tensor.exp(log_w)
        return w  


    def onehot(self,x,numclasses=10):
 
        if x.shape==():
            x = x[None]
        if numclasses is None:
            numclasses = x.max() + 1
        result = np.zeros(list(x.shape) + [numclasses], dtype="int32")
        z = np.zeros(x.shape)
        for c in range(numclasses):
            z *= 0
            z[np.where(x==c)] = 1
            result[...,c] += z
        return result
 

    def get_gradients(self, features, label, n_samples, mask):

        p_layers = self.p_layers 
        p_y = self.p_y
        p_h1 = self.p_h1 
 
        q_layers = self.q_layers
 
        q_y = self.q_y 
 
        q_y_given_x = self.q_y_given_x
        n_unsup_layers =  len(self.q_layers)
 
        batch_size = features.shape[0]
        
 
        x = replicate_batch(features, n_samples)
        y = replicate_batch(label, n_samples)
        log_q_y_given_x = q_y_given_x.log_prob( label, features)
 
 
        samples_layers_q_sup, samples_y_sup, log_p_layers_sup, log_p_y_sup ,log_p_h1_sup ,  log_q_layers_sup, log_q_y_sup = self.sample_q_sup ( x, y)
        
        w_sup = self.sup_importance_weights( log_p_layers_sup,log_p_y_sup ,log_p_h1_sup , log_q_layers_sup, log_q_y_sup,  batch_size, n_samples)
        
        # Calculate the accuracy of the model##########################################################################################
	"""
	zero_vector= np.zeros((100*10,10)).astype('int32')  
        all_lables = np.array([0,1,2,3,4,5,6,7,8,9]).reshape((1,10)).astype('int32') 
        

        all_lables = all_lables + zero_vector
        true_label = self.onehot(all_lables).astype('int32').reshape((1000,10,10))  
 
 
        class_prob =[]
        for i in range(10):
          sample_l = true_label[:,i,:].reshape((1000,10)) 
          c =  q_y.log_prob( sample_l, samples_layers_q_sup[n_unsup_layers])
          
          c = c.reshape((batch_size,n_samples))
          c= tensor.mean(c, axis=1)  
          c=  tensor.exp( c)
          class_prob.append( c )
    
        Classification_term = tensor.argmax(label, axis=1)- tensor.argmax(class_prob, axis=0)
        Classification_term = (1.0-(abs(Classification_term)>0.0)*1.0).mean()
	"""
        Classification_term = tensor.zeros([batch_size]).mean()

        # Sample from Q for unsupervise case ##########################################################################################
        samples_y_unsup,  log_p_layers_unsup, log_p_y_unsup ,log_p_h1_unsup , log_q_layers_unsup, log_q_y_unsup = self.sample_q_unsup ( x, n_samples, samples_layers_q_sup,y)
 
        
        w_unsup = self.unsup_importance_weights( log_p_layers_unsup,log_p_y_unsup ,log_p_h1_unsup , log_q_layers_unsup, log_q_y_unsup,  batch_size,n_samples)
              
         
        samples_layers_q_sup =  [v.reshape((batch_size,v.shape[1]*n_samples)) for v in samples_layers_q_sup]
 
        
        samples_y_sup = samples_y_sup.reshape((batch_size,samples_y_sup.shape[1]*n_samples))
       
        samples_y_unsup = samples_y_unsup.reshape((batch_size,samples_y_unsup.shape[1]*n_samples))
        
        log_p_layers_sup = [v.reshape((batch_size, n_samples)) for v in log_p_layers_sup]
        log_p_layers_unsup = [v.reshape((batch_size,n_samples)) for v in log_p_layers_unsup]
  
        log_p_h1_sup = log_p_h1_unsup.reshape((batch_size, n_samples)) 
        log_p_h1_unsup = log_p_h1_unsup.reshape((batch_size, n_samples)) 
        
        ########################################################################################################################################################################################################################################
         
        
        #samples_h2_layers = [tensor.switch(mask,u,v) for u, v in zip( samples_h2_layers_sup,samples_h2_layers_unsup)]
        samples_y  = tensor.switch(mask,samples_y_sup,samples_y_unsup)# [tensor.switch(mask,u,v) for u, v in zip( samples_y_sup,samples_y_unsup)]
        #samples_y = y
        #-------------------------   
        log_p_y_sup  = unflatten_values([log_p_y_sup], batch_size, n_samples)
        
        log_p_y_unsup  = unflatten_values([log_p_y_unsup], batch_size, n_samples)
        #log_p_y=   tensor.switch(mask,log_p_y_sup, log_p_y_unsup)
        #-------------------------   
        #log_p_layers = [tensor.switch(mask,u,v) for u, v in zip( log_p_layers_sup,log_p_layers_unsup)]
        #-------------------------   
        
        #log_p_h1=   tensor.switch(mask,log_p_h1_sup, log_p_h1_unsup)
        #-------------------------   sup_lower_bound
 
        #log_p_h2_layers   = [tensor.switch(mask,u,v) for u, v in zip( log_p_h2_layers_sup,log_p_h2_layers_unsup)]
        #-------------------------   
        log_q_layers_sup = [v.reshape((batch_size, n_samples)) for v in log_q_layers_sup]
        log_q_layers_unsup = [v.reshape((batch_size, n_samples)) for v in log_q_layers_unsup]
             
        #log_q_layers   = [tensor.switch(mask,u,v) for u, v in zip( log_q_layers_sup,log_q_layers_unsup)] 
        #-------------------------        
        
        #log_q_h2_layers   = [tensor.switch(mask,u,v) for u, v in zip( log_q_h2_layers_sup,log_q_h2_layers_unsup)] 
        
        #-------------------------
        log_q_y_sup  = unflatten_values([log_q_y_sup], batch_size, n_samples)
        log_q_y_unsup  = unflatten_values([log_q_y_unsup], batch_size, n_samples)
        #log_q_y=   tensor.switch(mask,log_q_y_sup, log_q_y_unsup)
 
       
        prob_layes_unsup = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers_unsup:
            prob_layes_unsup  =  p + prob_layes_unsup 
        
 
            
        prob_q_layes_unsup = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers_unsup:
            prob_q_layes_unsup=  p +   prob_q_layes_unsup

 
        
 
        fake_all_prob_unsup = 0.5*(prob_layes_unsup +log_p_y_unsup + log_p_h1_unsup -log_q_y_unsup - prob_q_layes_unsup   ) 
        unsup_lower_bound = 2.*logsumexp(fake_all_prob_unsup, axis=-1 ) -2.*tensor.log(n_samples )
        unsup_lower_bound = unsup_lower_bound.reshape((batch_size,1))
        
        
        #w_im =   tensor.switch(mask,w_sup, w_unsup) 
        w_im = w_sup
 
       
        prob_layes_sup = tensor.zeros([batch_size, n_samples])
        for p in log_p_layers_sup:
            prob_layes_sup  =  p + prob_layes_sup 
 
            
        prob_q_layes_sup = tensor.zeros([batch_size, n_samples])
        for p in log_q_layers_sup:
            prob_q_layes_sup=  p +   prob_q_layes_sup
 
        
        
     
        expectation =  log_q_y_unsup[0].reshape((batch_size, n_samples)).mean( axis=1).reshape((batch_size,1))
        expectation_sup =  log_q_y_sup[0].reshape((batch_size, n_samples)).mean( axis=1).reshape((batch_size,1))
   
        fake_all_prob_sup  =  0.5*(prob_layes_sup +  log_p_y_sup + log_p_h1_sup +log_q_y_sup - prob_q_layes_sup    )
        sup_lower_bound =  logsumexp( fake_all_prob_sup, axis=-1 ) -tensor.log(n_samples )     
        sup_lower_bound = sup_lower_bound.reshape((batch_size,1))
     
        alpha = 1.
        beta = 0.
        gamma = 0.
 
        total_lower_bound  = 0.5*unsup_lower_bound + sup_lower_bound  
        Objective_function= alpha * 0.5 * unsup_lower_bound + sup_lower_bound + beta * expectation + gamma * expectation_sup
 
        #all_prob = tensor.switch(mask, fake_all_prob_sup ,fake_all_prob_unsup )
   
        w = w_im.reshape( (batch_size*n_samples, ) )
        w_unsup =   w_unsup.reshape( (batch_size*n_samples, ) )
        w_unsup_q = w_unsup #- w_unsup.mean()
        w_sup =      w_sup.reshape( (batch_size*n_samples, ) )
        w_sup_q = w_sup#- w_sup.mean()
 
        samples_y  = flatten_values(samples_y , batch_size*n_samples)
        samples_y_unsup  = flatten_values(samples_y_unsup , batch_size*n_samples)
        samples_y_sup  = flatten_values(samples_y_unsup , batch_size*n_samples)
 
        #samples_h2_layers   =  [flatten_values(samples_h2_layers[v]   , batch_size*n_samples) for v in range(len(samples_h2_layers))]
 
        samples_layers_q_sup    = [flatten_values(samples_layers_q_sup[v]   , batch_size*n_samples) for v in range(len(samples_layers_q_sup))]
    
        gradients_unsup = OrderedDict()
        for l in xrange(n_unsup_layers ):
            gradients_unsup = merge_gradients(gradients_unsup, p_layers[l].get_gradients_list_h1( samples_layers_q_sup[l], samples_layers_q_sup[l+1], weights=[w_unsup,w_sup ], alpha = alpha , beta = beta ,gamma=gamma))
            gradients_unsup = merge_gradients(gradients_unsup, q_layers[l].get_gradients_list_h1( samples_layers_q_sup[l+1], samples_layers_q_sup[l], weights=[w_unsup_q ,w_sup_q  ], alpha = alpha , beta = beta ,gamma=gamma))
 
 
        gradients_unsup = merge_gradients(gradients_unsup, p_h1.get_gradients_list(samples_layers_q_sup[n_unsup_layers], samples_y_unsup,samples_layers_q_sup[n_unsup_layers], 
                                          samples_y_sup,weights=[w_unsup,w_sup ], alpha=alpha, beta=beta ,gamma=gamma))
 
        gradients_sup = OrderedDict()
  
        gradients_sup = merge_gradients(gradients_sup, p_y.get_gradients_list(samples_y_unsup,samples_y_sup, weights=[w_unsup,w_sup ],  alpha = alpha , beta = beta ,gamma=gamma))               
        gradients_sup = merge_gradients(gradients_sup, q_y.get_gradients_list_h1_type3(samples_y_unsup, samples_y_sup, samples_layers_q_sup[n_unsup_layers] , weights=[w_unsup   ,w_sup  ],  alpha = alpha , beta = beta ,gamma=gamma,expectation_term =True))   
    
 
        gradients  = OrderedDict()
        gradients = merge_gradients(gradients_unsup, gradients_sup) 
 
        return  expectation, Classification_term,log_q_y_sup[0], log_q_y_unsup[0], sup_lower_bound , 0.5*unsup_lower_bound, Objective_function, total_lower_bound, gradients 
