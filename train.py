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

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RemoveNotFinite, Momentum, RMSProp, Adam
from blocks.bricks import Tanh, Logistic, Rectifier
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
from helmholtz.bihm import BiHM
from helmholtz.dvae import DVAE
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.vae import VAE

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


def named(var, name):
    var.name = name
    return var

#-----------------------------------------------------------------------------
def vae_training():
    return cost, train_monitors, valid_monitors, test_monitors

#-----------------------------------------------------------------------------
def main(args):
    """Run experiment. """
    lr_tag = float_tag(args.learning_rate)

    x_dim, train_stream, valid_stream, test_stream = datasets.get_streams(args.data, args.batch_size)

    #------------------------------------------------------------
    # Setup model
    deterministic_act = Tanh
    deterministic_size = 1.

    if args.method == 'vae':
        sizes_tag = args.layer_spec.replace(",", "-")
        layer_sizes = [int(i) for i in args.layer_spec.split(",")]
        layer_sizes, z_dim = layer_sizes[:-1], layer_sizes[-1]

        name = "%s-%s-%s-lr%s-spl%d-%s" % \
            (args.data, args.method, args.name, lr_tag, args.n_samples, sizes_tag)

        if args.activation == "tanh":
            hidden_act = Tanh()
        elif args.activation == "logistic":
            hidden_act = Logistic()
        elif args.activation == "relu":
            hidden_act = Rectifier()
        else: 
            raise "Unknown hidden nonlinearity %s" % args.hidden_act

        model = VAE(x_dim=x_dim, hidden_layers=layer_sizes, hidden_act=hidden_act, z_dim=z_dim)
        model.initialize()
    elif args.method == 'dvae':
        sizes_tag = args.layer_spec.replace(",", "-")
        layer_sizes = [int(i) for i in args.layer_spec.split(",")]
        layer_sizes, z_dim = layer_sizes[:-1], layer_sizes[-1]

        name = "%s-%s-%s-lr%s-spl%d-%s" % \
            (args.data, args.method, args.name, lr_tag, args.n_samples, sizes_tag)

        if args.activation == "tanh":
            hidden_act = Tanh()
        elif args.activation == "logistic":
            hidden_act = Logistic()
        elif args.activation == "relu":
            hidden_act = Rectifier()
        else: 
            raise "Unknown hidden nonlinearity %s" % args.hidden_act

        model = DVAE(x_dim=x_dim, hidden_layers=layer_sizes, hidden_act=hidden_act, z_dim=z_dim)
        model.initialize()
    elif args.method == 'rws':
        sizes_tag = args.layer_spec.replace(",", "-")
        name = "%s-%s-%s-lr%s-dl%d-spl%d-%s" % \
            (args.data, args.method, args.name, lr_tag, args.deterministic_layers, args.n_samples, sizes_tag)

        p_layers, q_layers = create_layers(
                                args.layer_spec, x_dim,
                                args.deterministic_layers,
                                deterministic_act, deterministic_size)

        model = ReweightedWakeSleep(
                p_layers,
                q_layers,
            )
        model.initialize()
    elif args.method == 'bihm-rws':
        sizes_tag = args.layer_spec.replace(",", "-")
        name = "%s-%s-%s-lr%s-dl%d-spl%d-%s" % \
            (args.data, args.method, args.name, lr_tag, args.deterministic_layers, args.n_samples, sizes_tag)

        p_layers, q_layers = create_layers(
                                args.layer_spec, x_dim,
                                args.deterministic_layers,
                                deterministic_act, deterministic_size)

        model = BiHM(
                p_layers,
                q_layers,
                baseline=args.qbaseline
            )
        model.initialize()
    elif args.method == 'continue':
        import cPickle as pickle
        from os.path import basename, splitext


        with open(args.model_file, 'rb') as f:
            m = pickle.load(f)

        if isinstance(m, MainLoop):
            m = m.model

        model = m.get_top_bricks()[0]
        while len(model.parents) > 0:
            model = model.parents[0]

        assert isinstance(model, (BiHM, ReweightedWakeSleep, DVAE, VAE))

        fname = basename(args.model_file)
        mname = fname.rpartition("_model.pkl")
        name = "%s-cont-%s-lr%s-spl%s" % (mname, args.name, lr_tag, args.n_samples)
    else:
        raise ValueError("Unknown training method '%s'" % args.method)

    #------------------------------------------------------------

    x = tensor.matrix('features')

    #------------------------------------------------------------
    # Testset monitoring

    train_monitors = []
    valid_monitors = []
    test_monitors = []
    for s in [1, 10, 100, 1000]:
        log_p, log_ph = model.log_likelihood(x, s)
        log_p  = -log_p.mean()
        log_ph = -log_ph.mean()
        log_p.name  = "log_p_%d" % s
        log_ph.name = "log_ph_%d" %s

        #train_monitors += [log_p, log_ph]
        #valid_monitors += [log_p, log_ph]
        test_monitors += [log_p, log_ph]

    #------------------------------------------------------------
    # Gradient and training monitoring

    if args.method in ['vae', 'dvae']:
        log_p_bound = model.log_likelihood_bound(x, args.n_samples)
        gradients = None
        log_p_bound  = -log_p_bound.mean()
        log_p_bound.name  = "log_p_bound"
        cost = log_p_bound

        train_monitors += [log_p_bound, named(model.kl_term.mean(), 'kl_term'), named(model.recons_term.mean(), 'recons_term')]
        valid_monitors += [log_p_bound, named(model.kl_term.mean(), 'kl_term'), named(model.recons_term.mean(), 'recons_term')]
        test_monitors  += [log_p_bound, named(model.kl_term.mean(), 'kl_term'), named(model.recons_term.mean(), 'recons_term')]
    else:
        log_p, log_ph, gradients = model.get_gradients(x, args.n_samples)
        log_p  = -log_p.mean()
        log_ph = -log_ph.mean()
        log_p.name  = "log_p"
        log_ph.name = "log_ph"
        cost = log_ph

        train_monitors += [log_p, log_ph]
        valid_monitors += [log_p, log_ph]

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
    cg = ComputationGraph([cost])

    #------------------------------------------------------------

    if args.step_rule == "momentum":
        step_rule = Momentum(args.learning_rate, 0.95)
    elif args.step_rule == "rmsprop":
        step_rule = RMSProp(args.learning_rate)
    elif args.step_rule == "adam":
        step_rule = Adam(args.learning_rate)
    else:
        raise "Unknown step_rule %s" % args.step_rule

    #parameters = cg.parameters[:4] + cg.parameters[5:]
    parameters = cg.parameters
    print(parameters)

    algorithm = GradientDescent(
        cost=cost,
        parameters=parameters,
        gradients=gradients,
        step_rule=CompositeRule([
            StepClipping(25),
            step_rule,
            RemoveNotFinite(0.9),
        ])
    )

    #------------------------------------------------------------

    train_monitors += [aggregation.mean(algorithm.total_gradient_norm),
                       aggregation.mean(algorithm.total_step_norm)]

    #------------------------------------------------------------


    # Live plotting?
    plotting_extensions = []
    if args.live_plotting:
        plotting_extensions = [
            PlotManager(
                name,
                [Plotter(channels=[
                        ["valid_%s" % cost.name, "valid_log_p"],
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
        model=Model(cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[Timing(),
                    ProgressBar(),
                    TrainingDataMonitoring(
                        train_monitors,
                        prefix="train",
                        after_epoch=True),
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
                    TrackTheBest('valid_%s' % cost.name),
                    Checkpoint(name+".pkl", save_separately=['log', 'model']),
                    FinishIfNoImprovementAfter('valid_%s_best_so_far' % cost.name, epochs=args.patience),
                    FinishAfter(after_n_epochs=args.max_epochs),
                    Printing()] + plotting_extensions)
    main_loop.run()

#=============================================================================

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default="", help="Name for this experiment")
    parser.add_argument("--data", "-d", dest='data', choices=datasets.supported_datasets,
                default='bmnist', help="Dataset to use")
    parser.add_argument("--live-plotting", "--plot", action="store_true", default=False,
                help="Enable live plotting to a Bokkeh server")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch (default: 100)")
    parser.add_argument("--step-rule", choices=['momentum', 'adam', 'rmsprop'], dest="step_rule",
                default="adam", help="Chose SGD alrogithm (default: adam)"),
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--max-epochs", "--epochs", type=int, dest="max_epochs",
                default=10000, help="Maximum # of training epochs to run")
    parser.add_argument("--early-stopping", type=int, dest="patience", 
                default=10, help="Number of epochs without improvement (default: 10)")
    subparsers = parser.add_subparsers(title="methods", dest="method")

    # Continue
    subparser = subparsers.add_parser("continue",
                help="Variational Auto Encoder with Gaussian latents and Bernoulli observed")
    subparser.add_argument("model_file", type=str, 
                help="Model .pkl lto load and continue")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of samples")

    # Variational Autoencoder
    subparser = subparsers.add_parser("vae",
                help="Variational Auto Encoder with Gaussian latents and Bernoulli observed")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=1, help="Number of samples")
    subparser.add_argument("--activation", choices=['tanh', 'logistic', 'relu'], dest="activation",
                default='relu', help="Activation function (last p(x|z) layer is always Logistic; default: relu)")
    subparser.add_argument("layer_spec", type=str, 
                default="200,100", help="Comma seperated list of layer sizes (last is z-dim)")

    # Descrete Variational Autoencoder
    subparser = subparsers.add_parser("dvae",
                help="Discrete Variational Auto Encoder with Bernoulli latents and observed")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=1, help="Number of samples")
    subparser.add_argument("--activation", choices=['tanh', 'logistic', 'relu'], dest="activation",
                default='relu', help="Activation function (last p(x|z) layer is always Logistic; default: relu)")
    subparser.add_argument("layer_spec", type=str, 
                default="200,100", help="Comma seperated list of layer sizes (last is z-dim)")

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
    subparser.add_argument("--qbaseline", choices=['none', 'datapoint', 'batch'], 
                default='none', help="Use a baseline for Q updates")
    subparser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="Deterministic hidden layers per stochastic layer")
    subparser.add_argument("layer_spec", type=str,
                default="200,200,200", help="Comma seperated list of layer sizes")

    args = parser.parse_args()

    main(args)
