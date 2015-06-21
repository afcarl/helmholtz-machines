#!/usr/bin/env python2

from __future__ import division, print_function

import sys
sys.path.append("../")
sys.setrecursionlimit(100000)

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import ipdb
import fuel
import theano
import numpy as np

import blocks
import blocks.extras

from argparse import ArgumentParser
from collections import OrderedDict

from theano import tensor
from theano import tensor as T

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, Scale, Momentum, BasicMomentum, RMSProp, StepClipping, Adam, RemoveNotFinite
from blocks.bricks import Tanh, Logistic
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.main_loop import MainLoop
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER
from blocks.utils import shared_floatx

import helmholtz.datasets as datasets

from helmholtz import create_layers
from helmholtz.gmm import GMM
from helmholtz.rws import ReweightedWakeSleep

floatX = theano.config.floatX
fuel.config.floatX = floatX

def half_lr_func(niter, lr):
    return np.cast[floatX](lr/2.)

def float_tag(value):
    """ Convert a float into a short tag-usable string representation. E.g.:
        0.1   -> 11
        0.01  -> 12
        0.001 -> 13
        0.005 -> 53
    """
    if value == 0.0:
        return "00"
    exp = np.floor(np.log10(value))
    leading = ("%e"%value)[0]
    return "%s%d" % (leading, -exp)


#-----------------------------------------------------------------------------
def main(data, batch_size, learning_rate, epochs, n_samples, layer_spec, deterministic_layers, name):
    """ Run experiment
    """
    lr_tag = float_tag(learning_rate)
    sizes_tag = layer_spec.replace(",", "-")

    name = "%s-%s-lr%s-e%d-hl%d-spl%d-%s" % (data, name, lr_tag, epochs, deterministic_layers, n_samples, sizes_tag)

    half_lr = 100 

    #------------------------------------------------------------
    
    x_dim, data_train, data_valid, data_test = datasets.get_data(data)

    #------------------------------------------------------------
    # Setup model 

    deterministic_act = Tanh
    deterministic_size = 1.

    p_layers, q_layers = create_layers(layer_spec, x_dim, deterministic_layers, deterministic_act, deterministic_size)

    model = ReweightedWakeSleep(
            p_layers, 
            q_layers, 
        )
    model.initialize()

    #------------------------------------------------------------

    x = tensor.matrix('features')

    #------------------------------------------------------------
    # Testset monitoring

    test_monitors = []
    for s in [1, 10, 50, 100]:
        log_p, log_ph = model.log_likelihood(x, s)
        log_p  = -log_p.mean()
        log_ph = -log_ph.mean()
        log_p.name  = "log_p_%d" % s
        log_ph.name = "log_ph_%d" %s

        test_monitors.append(log_p)
        test_monitors.append(log_ph)
 

    #------------------------------------------------------------
    # Gradient and training monitoring

    log_p, log_ph, gradients = model.get_gradients(x, n_samples)
    log_p  = -log_p.mean()
    log_ph = -log_ph.mean()
    log_p.name  = "log_p"
    log_ph.name = "log_ph"

    train_monitors = [log_p, log_ph]
    valid_monitors = [log_p, log_ph]
   
    #------------------------------------------------------------
    # Detailed monitoring
    """
    n_layers = len(p_layers)
    
    log_px, w, log_p, log_q, samples = model.log_likelihood(x, n_samples)

    exp_samples = []
    for l in xrange(n_layers):
        e = (w.dimshuffle(0, 1, 'x')*samples[l]).sum(axis=1)
        e.name = "inference_h%d" % l
        e.tag.aggregation_scheme = aggregation.TakeLast(e)
        exp_samples.append(e)
    
    s1 = samples[1]
    sh1 = s1.shape
    s1_ = s1.reshape([sh1[0]*sh1[1], sh1[2]])
    s0, _ = model.p_layers[0].sample_expected(s1_)
    s0 = s0.reshape([sh1[0], sh1[1], s0.shape[1]])
    s0 = (w.dimshuffle(0, 1, 'x')*s0).sum(axis=1)
    s0.name = "inference_h0^"
    s0.tag.aggregation_scheme = aggregation.TakeLast(s0)
    exp_samples.append(s0)


    # Draw P-samples
    p_samples, _, _ = model.sample_p(100)
    #weights = model.importance_weights(samples)
    #weights = weights / weights.sum()

    for i, s in enumerate(p_samples):
        s.name = "psamples_h%d" % i
        s.tag.aggregation_scheme = aggregation.TakeLast(s)
    
    #
    samples = model.sample(100, oversample=100)

    for i, s in enumerate(samples):
        s.name = "samples_h%d" % i
        s.tag.aggregation_scheme = aggregation.TakeLast(s)
    """
    cg = ComputationGraph([log_ph])

    #------------------------------------------------------------

    algorithm = GradientDescent(
        cost=log_ph,
        gradients=gradients,
        step_rule=CompositeRule([
            RMSProp(learning_rate),
            #Adam(learning_rate),
            RemoveNotFinite(0.9),
        ])

#        step_rule=Adam(learning_rate),
#        step_rule=Momentum(learning_rate, 0.95),
#        step_rule=CompositeRule([
#                    StepClipping(1),
#                    RMSProp(learning_rate),
#                    Adam(learning_rate),
#                    Scale(learning_rate=learning_rate),
#                    BasicMomentum(0.95)),
#        ])
    )

    #------------------------------------------------------------

    train_monitors += [aggregation.mean(algorithm.total_gradient_norm),
                       aggregation.mean(algorithm.total_step_norm)]

 
    #------------------------------------------------------------

    train_stream = Flatten(
        DataStream(
            data_train,
            iteration_scheme=ShuffledScheme(data_train.num_examples, batch_size)
        ), 
        which_sources='features')
    valid_stream = Flatten(
        DataStream(
            data_valid,
            iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)
        ),
        which_sources='features')
    test_stream = Flatten(
        DataStream(
            data_test,
            iteration_scheme=SequentialScheme(data_test.num_examples, batch_size)
        ),
        which_sources='features')


    main_loop = MainLoop(
        model=Model(log_ph),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[Timing(),
                    ProgressBar(),
                    FinishAfter(after_n_epochs=epochs),
                    TrainingDataMonitoring(
                        train_monitors,
                        prefix="train",
                        after_batch=True),
                    DataStreamMonitoring(
                        valid_monitors, 
                        data_stream=valid_stream,
                        prefix="valid"),
                    DataStreamMonitoring(
                        test_monitors, 
                        data_stream=test_stream,
                        prefix="test",
                        after_epoch=False,
                        after_training=True,
                        every_n_epochs=10),
                    #SharedVariableModifier(
                    #    algorithm.step_rule.components[0].learning_rate,
                    #    half_lr_func,
                    #    before_training=False,
                    #    after_epoch=False,
                    #    after_batch=False,
                    #    every_n_epochs=half_lr),
                    TrackTheBest('valid_log_p'),
                    TrackTheBest('valid_log_ph'),
                    Checkpoint(name+".pkl", save_separately=['log', 'model']),
                    Plot(
                        name,
                        channels=[
                            #["train_log_ph", "train_log_p", "valid_log_ph", "valid_log_p", "test_log_ph_100", "test_log_p_100"], 
                            ["valid_log_ph", "valid_log_p"], 
                            ["train_total_gradient_norm", "train_total_step_norm"]
                        ]),
                   Printing()])
    main_loop.run()

#=============================================================================

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=50, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of IS samples")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="hidden layers per stochastic layer")
    parser.add_argument("--name", type=str, dest="name",
                default="", help="Name for this experiment")
    parser.add_argument("layer_spec", type=str, 
                default="200,200,200", help="Comma seperated list of layer sizes")
    args = parser.parse_args()

    main(**vars(args))
