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

import blocks.extras
import blocks

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
from blocks.extensions.stopping import FinishIfNoImprovementAfter
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

from blocks_extras.extensions.plot import PlotManager, Plotter, DisplayImage
from blocks_extras.extensions.display import ImageDataStreamDisplay, WeightDisplay, ImageSamplesDisplay

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
def main(args):
    """Run experiment. """
    lr_tag = float_tag(args.learning_rate)
    sizes_tag = args.layer_spec.replace(",", "-")

    name = "%s-%s-%s-lr%s-dl%d-spl%d-%s" % \
            (args.method, args.data, args.name, lr_tag, args.deterministic_layers, args.n_samples, sizes_tag)

    #half_lr = 100

    #------------------------------------------------------------

    x_dim, data_train, data_valid, data_test = datasets.get_data(args.data)

    #------------------------------------------------------------
    # Setup model
    deterministic_act = Tanh
    deterministic_size = 1.

    p_layers, q_layers = create_layers(args.layer_spec, x_dim, args.deterministic_layers, deterministic_act, deterministic_size)


    if args.method == 'rws':
        model = ReweightedWakeSleep(
                p_layers,
                q_layers,
            )
    elif args.method == 'bihm-rws':
        model = GMM(
                p_layers,
                q_layers,
            )
    else:
        raise ValueError("Unknown training method '%s'" % args.method)

    model.initialize()

    #------------------------------------------------------------

    x = tensor.matrix('features')

    #------------------------------------------------------------
    # Testset monitoring

    test_monitors = []
    for s in [1, 10, 100, 1000]:
        log_p, log_ph = model.log_likelihood(x, s)
        log_p  = -log_p.mean()
        log_ph = -log_ph.mean()
        log_p.name  = "log_p_%d" % s
        log_ph.name = "log_ph_%d" %s

        test_monitors.append(log_p)
        test_monitors.append(log_ph)
 

    #------------------------------------------------------------
    # Gradient and training monitoring

    log_p, log_ph, gradients = model.get_gradients(x, args.n_samples)
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
            Adam(args.learning_rate),
#            RMSProp(args.learning_rate),
#            Momentum(args.learning_rate, 0.95),
            RemoveNotFinite(0.9),
        ])
    )

    #------------------------------------------------------------

    train_monitors += [aggregation.mean(algorithm.total_gradient_norm),
                       aggregation.mean(algorithm.total_step_norm)]

    #------------------------------------------------------------

    # Out usual train/valid/test data streams...
    train_stream, valid_stream, test_stream = (
            Flatten(DataStream(
                data,
                iteration_scheme=ShuffledScheme(data.num_examples, batch_size)
            ), which_sources='features')
        for data, batch_size in ((data_train, args.batch_size),
                                 (data_valid, args.batch_size//2),
                                 (data_test, args.batch_size//2))
    )

    # A single datapooint per for detailed gradient monitoring...
    gradient_stream = Flatten(
        DataStream(
            data_train,
            iteration_scheme=ShuffledScheme(data_train.num_examples, args.batch_size)
        ),
        which_sources='features')
    valid_stream = Flatten(
        DataStream(
            data_valid,
            iteration_scheme=SequentialScheme(data_valid.num_examples, args.batch_size)
        ),
        which_sources='features')
    test_stream = Flatten(
        DataStream(
            data_test,
            iteration_scheme=SequentialScheme(data_test.num_examples, args.batch_size)
        ),
        which_sources='features')

    plotting_extensions = []
    if args.live_plotting:
        plotting_extensions = [
            PlotManager(
                name,
                [Plotter(channels=[
                        ["valid_log_ph", "valid_log_p"],
                        ["train_total_gradient_norm", "train_total_step_norm"]],
                    titles=[
                        "validation cost",
                        "norm of training gradient and step"
                    ]),
                DisplayImage([
                    WeightDisplay(
                        model.p_layers[0].mlp.linear_transformations[0].W,
                        n_weights=100, image_shape=(28, 28))]
                    #ImageDataStreamDisplay(test_stream, image_shape=(28,28))]
                )]
            )
        ]

    main_loop = MainLoop(
        model=Model(log_ph),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[Timing(),
                    ProgressBar(),
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
                    FinishIfNoImprovementAfter('valid_log_p_best_so_far', epochs=10),
                    FinishAfter(after_n_epochs=args.max_epochs),
                    Printing()] + plotting_extensions)
    main_loop.run()

#=============================================================================

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--live-plotting", "--plot", action="store_true", default=False,
                help="Enable live plotting to a Bokkeh server")
    parser.add_argument("--max-epochs", "--epochs", type=int, dest="max_epochs",
                default=10000, help="Maximum # of training epochs to run")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--name", type=str, dest="name",
                default="", help="Name for this experiment")
    subparsers = parser.add_subparsers(title="methods", dest="method")

    # Reweighted Wake-Sleep
    subparser = subparsers.add_parser("rws",
                help="Reweighted Wake-Sleep")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of IS samples")
    subparser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="Deterministic hidden layers per stochastic layer")
    subparser.add_argument("layer_spec", type=str,
                default="200,200,200", help="Comma seperated list of layer sizes")

    # Bidirection HM
    subparser = subparsers.add_parser("bihm-rws",
                help="Bidirectional Helmholtz Machine with RWS")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of IS samples")
    subparser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="Deterministic hidden layers per stochastic layer")
    subparser.add_argument("layer_spec", type=str,
                default="200,200,200", help="Comma seperated list of layer sizes")

    args = parser.parse_args()

    main(args)
