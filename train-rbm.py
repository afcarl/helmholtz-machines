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

#import blocks.extras
import blocks

from argparse import ArgumentParser
from collections import OrderedDict

from theano import tensor
from theano import tensor as T

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RemoveNotFinite, Momentum, RMSProp, Adam, Restrict
from blocks.bricks import Tanh, Logistic, Rectifier
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.main_loop import MainLoop
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER
from blocks.select import Selector
from blocks.utils import shared_floatx

from blocks_extras.extensions.plot import PlotManager, Plotter, DisplayImage
from blocks_extras.extensions.display import ImageDataStreamDisplay, WeightDisplay, ImageSamplesDisplay

import helmholtz.datasets as datasets

from helmholtz import create_layers
from helmholtz.rws import ReweightedWakeSleep

#from helmholtz.nade import NADETopLayer
from helmholtz.rbm import RBMTopLayer

est_z = __import__("est-z-rbm")
ComputeLogZ = est_z.ComputeLogZ

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
def main(args):
    """Run experiment. """
    if args.learning_rate_rbm is None:
        args.learning_rate_rbm = args.learning_rate

    lr_tag  = float_tag(args.learning_rate)
    lrr_tag = float_tag(args.learning_rate_rbm)

    x_dim, train_stream, valid_stream, test_stream = datasets.get_streams(args.data, args.batch_size)

    #------------------------------------------------------------
    # Setup model
    deterministic_act = Tanh
    deterministic_size = 1.

    if args.method == 'rws':
        sizes_tag = args.layer_spec.replace(",", "-")
        qbase = "" if not args.no_qbaseline else "noqb-"

        if args.pcd_training:
            train_tag = "pcd"
        else:
            train_tag = "cd%d" % args.cd_iterations

        name = "%s-%s-rbm-%s-%s--h%d-%slr%s-lrr%s-dl%d-spl%d-%s" % \
            (args.data, args.method, args.name, train_tag, args.rbm_hiddens, qbase, lr_tag, lrr_tag, args.deterministic_layers, args.n_samples, sizes_tag)

        p_layers, q_layers = create_layers(
                                args.layer_spec, x_dim,
                                args.deterministic_layers,
                                deterministic_act, deterministic_size)

        # Replace top level with something more interesting!
        top_size = p_layers[-1].dim_X
        rbm = RBMTopLayer(
                        dim_x=top_size,
                        dim_h=args.rbm_hiddens,
                        cd_iterations=args.cd_iterations,
                        pcd_training=args.pcd_training,
                        name="p_top_rbm",
                        weights_init=IsotropicGaussian(std=0.01),
                        biases_init=Constant(0.0))
        p_layers[-1] = rbm

        model = ReweightedWakeSleep(
                p_layers,
                q_layers,
                qbaseline=(not args.no_qbaseline),
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

        assert isinstance(model, (BiHM, ReweightedWakeSleep, VAE))

        mname, _, _ = basename(args.model_file).rpartition("_model.pkl")
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
        log_p, _ = model.log_likelihood(x, s)
        log_p = named(-log_p.mean(), "unnorm_log_p_%d" % s)

        valid_monitors += [log_p]
        test_monitors += [log_p]

    #------------------------------------------------------------
    # Gradient and training monitoring

    log_p, _, gradients = model.get_gradients(x, args.n_samples)
    log_p  = named(-log_p.mean(), "unnorm_log_p")
    cost = log_p

    train_monitors += [log_p]
    valid_monitors += [log_p]

    #------------------------------------------------------------

    if args.step_rule == "momentum":
        step_rule = Momentum(args.learning_rate, 0.95)
        step_rule_rbm = Momentum(args.learning_rate_rbm, 0.95)
    elif args.step_rule == "rmsprop":
        step_rule = RMSProp(args.learning_rate)
        step_rule_rbm = RMSProp(args.learning_rate_rbm)
    elif args.step_rule == "adam":
        step_rule = Adam(args.learning_rate)
        step_rule_rbm = Adam(args.learning_rate_rbm)
    else:
        raise "Unknown step_rule %s" % args.step_rule

    cg = ComputationGraph([cost])
    parameters = cg.parameters

    step_rule_rbm_params = Selector(model.p_layers[-1]).get_parameters().values()
    step_rule_params = [p for p in parameters if p not in step_rule_rbm_params]

    algorithm = GradientDescent(
        cost=cost,
        parameters=parameters,
        gradients=gradients,
        step_rule=CompositeRule([
            #StepClipping(25),
            #step_rule,
            Restrict(step_rule, step_rule_params),
            Restrict(step_rule_rbm, step_rule_rbm_params),
            #RemoveNotFinite(1.0),
        ])
    )
    algorithm.add_updates(rbm.pcd_updates)

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
                        prefix="valid",
                        after_epoch=False,
                        after_training=True,
                        every_n_epochs=10),
                    DataStreamMonitoring(
                        test_monitors,
                        data_stream=test_stream,
                        prefix="test",
                        after_epoch=False,
                        after_training=True,
                        every_n_epochs=10),
                    #ComputeLogZ(
                    #    rbm=rbm,
                    #    every_n_epochs=10),
                    TrackTheBest('valid_log_p'),
                    TrackTheBest('valid_log_p_1000'),
                    Checkpoint(name+".pkl", save_separately=['log', 'model']),
                    FinishIfNoImprovementAfter('valid_log_p_best_so_far', epochs=args.patience),
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
    parser.add_argument("--lr-rbm", "--learning-rate-rbm", type=float, dest="learning_rate_rbm",
                default=None, help="Learning rate")
    parser.add_argument("--max-epochs", "--epochs", type=int, dest="max_epochs",
                default=10000, help="Maximum # of training epochs to run")
    parser.add_argument("--early-stopping", type=int, dest="patience",
                default=20, help="Number of epochs without improvement (default: 10)")
    subparsers = parser.add_subparsers(title="methods", dest="method")

    # Continue
    subparser = subparsers.add_parser("continue",
                help="Variational Auto Encoder with Gaussian latents and Bernoulli observed")
    subparser.add_argument("model_file", type=str,
                help="Model .pkl lto load and continue")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of samples")

    # Reweighted Wake-Sleep
    subparser = subparsers.add_parser("rws",
                help="Reweighted Wake-Sleep")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of IS samples")
    subparser.add_argument("--cd-iterations", "--cd", type=int, dest="cd_iterations",
                default=10, help="Number of CD iterations")
    subparser.add_argument("--pcd-training", "--pcd", action="store_true", 
                help="Use PCD for training")
    subparser.add_argument("--rbm-hiddens", type=int, dest="rbm_hiddens",
                default=20, help="Number RBM hidden units")
    subparser.add_argument("--no-qbaseline", "--nobase", action="store_true",
                default=False, help="Deactivate 1/n_samples baseline for Q gradients")
    subparser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="Deterministic hidden layers per stochastic layer")
    subparser.add_argument("layer_spec", type=str,
                default="200,200,200", help="Comma seperated list of layer sizes")

    args = parser.parse_args()

    main(args)
