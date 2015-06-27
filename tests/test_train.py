
import unittest 

import argparse
import train


@unittest.skip("takes too long")
def test_main():
    args = argparse.Namespace
    args.data = "bmnist"
    args.live_plotting = False
    args.max_epochs = 1
    args.batch_size = 100
    args.learning_rate = 1e-3
    args.name = "nosetest"
    args.method = "rws"
    args.n_samples = 10
    args.deterministic_layers = 0
    args.layer_spec = "10,5"

    train.main(args)
