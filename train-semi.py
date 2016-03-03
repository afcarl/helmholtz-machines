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

from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RemoveNotFinite, Momentum, RMSProp, Adam
from blocks.bricks import MLP, Tanh, Logistic, Rectifier
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

#import helmholtz.datasets as datasets
import helmholtz.semi_datasets as semi_datasets

from helmholtz import create_layers
from helmholtz.bihm import BiHM
from helmholtz.semibihm import SemiBiHM
from helmholtz.dvae import DVAE
from helmholtz.rws import ReweightedWakeSleep
from helmholtz.vae import VAE

from helmholtz.prob_layers import MultinomialLayer, MultinomialTopLayer
from blocks.initialization import Uniform, IsotropicGaussian, Constant, Sparse

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
    lr_tag = float_tag(args.learning_rate)

    # Batch size
    bs_labeled, bs_unlabeled = [int(i) for i in args.batch_size.split(",")]

    import ipdb; ipdb.set_trace()

    x_dim, y_dim, train_stream, valid_stream, test_stream = semi_datasets.get_streams(args.data, 1000, (bs_labeled, bs_unlabeled))

    #------------------------------------------------------------
    # Setup model
    deterministic_act = Tanh
    deterministic_size = 1.

    if args.method == 'semi-bihm':
        sizes_tag = args.layer_spec.replace(",", "-")
        name = "%s-%s-%s-bs%d%d-lr%s-dl%d-spl%d-%s" % \
            (args.data, args.method, args.name, bs_labeled, bs_unlabeled, lr_tag, args.deterministic_layers, args.n_samples, sizes_tag)

        # create SBN layers
        bottom_p, bottom_q = create_layers(
                                args.layer_spec, x_dim,
                                args.deterministic_layers,
                                deterministic_act, deterministic_size)

        # replace the top-most Q layer with a conditional categorical distribution
        inits = {
            'weights_init': IsotropicGaussian(),
            'biases_init': Constant(0.0),
        }
        bottom_q[-1] = MultinomialLayer(MLP([Logistic()], [25, 10], **inits))

        top_p = MultinomialTopLayer(10, **inits)  # create top level P prior
        #top_p = bottom_p[-1]
        del bottom_p[-1]                          # remove the top layer P layer


        model = SemiBiHM(bottom_p, bottom_q, top_p)
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

        assert isinstance(model, (SemiBiHM, ))

        model.px_weight = args.px_weight
            


        mname, _, _ = basename(args.model_file).rpartition("_model.pkl")
        name = "%s-cont-%s-lr%s-spl%s" % (mname, args.name, lr_tag, args.n_samples)
    else:
        raise ValueError("Unknown training method '%s'" % args.method)

    #------------------------------------------------------------

    mask = tensor.imatrix('mask')
    #mask = tensor.zeros((args.batch_size,1)).astype('int32')# tensor.col('mask', dtype='int32')

    x = tensor.matrix('features')
    #x = x.reshape((args.batch_size,x_dim ))
    y = tensor.matrix('targets')
    #y = y.reshape((args.batch_size,y_dim ))


    #------------------------------------------------------------
    # Testset monitoring

    train_monitors = []
    valid_monitors = []
    test_monitors = []
    for s in [1, 10, 100, ]:
        log_px, log_pxy, log_pygx = model.log_likelihood(x, y, mask, s)
        #log_px, log_pygx = model.log_likelihood(x, s)

        log_px   = named(-log_px.mean(), "log_px_%d" % s)
        log_pxy  = named(-log_pxy.sum() / mask.sum(), "log_pxy_%d" % s)
        log_pygx = named(-log_pygx.sum() / mask.sum(), "log_pygx_%d" % s)

        test_monitors += [log_px, log_pxy, log_pygx]

	### Jorgs old stuff ###
        # log_p, log_ph = model.log_likelihood(x, s)
        # log_p  = -log_p.mean()
        # log_ph = -log_ph.mean()
        # log_p.name  = "log_p_%d" % s
        # log_ph.name = "log_ph_%d" % s
        # test_monitors += [log_p, log_ph]

    #------------------------------------------------------------
    # Gradient and training monitoring

 
    gradients, log_px, log_pxy, log_pygx = model.get_gradients(x, y, mask, args.n_samples)

    log_px   = named(-log_px.mean(), "log_px")
    log_pxy  = named(-log_pxy.sum() / mask.sum(), "log_pxy")
    log_pygx = named(-log_pygx.sum() / mask.sum(), "log_pygx")
    cost = log_pygx

    train_monitors = [log_px, log_pxy, log_pygx]
    valid_monitors = [log_px, log_pxy, log_pygx]


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
    #parameters = cg.parameters

    algorithm = GradientDescent(
        cost=cost,
        gradients=gradients,
        step_rule=CompositeRule([
            #StepClipping(25),
            step_rule,
            #RemoveNotFinite(1.0),
        ]),
        on_unused_sources='warn'
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
                        ["train_lower_bound" ,"valid_lower_bound" ],
                        ["train_total_gradient_norm", "train_total_step_norm"]],
                    titles=[
                        "validation cost",
                        "more stuff",
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
                    TrackTheBest('train_%s' % cost.name),
                    Checkpoint(name+".pkl", save_separately=['log', 'model'], use_cpickle=True),
                    FinishIfNoImprovementAfter('valid_%s_best_so_far' % cost.name, epochs=args.patience),
                    FinishAfter(after_n_epochs=args.max_epochs),
                    Printing()] + plotting_extensions)
    main_loop.run()

#=============================================================================

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default="", help="Name for this experiment")
    parser.add_argument("--data", "-d", dest='data', choices=semi_datasets.supported_datasets,
                default='mnist', help="Dataset to use")
    parser.add_argument("--live-plotting", "--plot", action="store_true", default=False,
                help="Enable live plotting to a Bokkeh server")
    parser.add_argument("--bs", "--batch-size", type=str, dest="batch_size",
                default="50,50", help="Comma seperated batch size for labeled and unlabeled data (default: 50,50)")
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
    subparser.add_argument("--px-weight", type=float, dest="px_weight",
                default=1.0, help="Weight for log p*(x)")

    # Semisupervised Bidirection HM
    subparser = subparsers.add_parser("semi-bihm",
                help="Semisupervise BiHM with RWS")
    subparser.add_argument("--nsamples", "-s", type=int, dest="n_samples",
                default=10, help="Number of IS samples")
    subparser.add_argument("--deterministic-layers", type=int, dest="deterministic_layers",
                default=0, help="Deterministic hidden layers per stochastic layer")
    subparser.add_argument("--supervised_layer_size", type=str,  dest="supervised_layer_size",
                default="", help="Comma seperated list of supervised layer sizes")
    subparser.add_argument("layer_spec", type=str,
                default="200,200,200", help="Comma seperated list of layer sizes")

    args = parser.parse_args()

    main(args)
