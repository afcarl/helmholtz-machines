
from __future__ import division


local_datasets = ["adult", "dna", "web", "nips", "mushrooms", "ocr_letters", "connect4", "rcv1", "silhouettes"]
supported_datasets = local_datasets + ['bmnist', 'tfd', 'bars']

def get_data(data_name):
    if data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        x_dim = 28*28 

        data_train = BinarizedMNIST(which_sets=['train'], sources=['features'])
        data_valid = BinarizedMNIST(which_sets=['valid'], sources=['features'])
        data_test  = BinarizedMNIST(which_sets=['test'], sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase

        x_dim = 48*48

        data_train = TorontoFaceDatabase(which_sets=['unlabeled'], sources=['features'])
        data_valid = TorontoFaceDatabase(which_sets=['valid'], sources=['features'])
        data_test  = TorontoFaceDatabase(which_sets=['test'], sources=['features'])
    elif data_name == 'bars':
        from bars_data import Bars

        width = 4
        x_dim = width*width

        data_train = Bars(num_examples=5000, width=width, sources=['features'])
        data_valid = Bars(num_examples=5000, width=width, sources=['features'])
        data_test  = Bars(num_examples=5000, width=width, sources=['features'])
    elif data_name in local_datasets:
        from fuel.datasets.hdf5 import H5PYDataset

        fname = "../data/"+data_name+".hdf5"
        
        data_train = H5PYDataset(fname, which_sets=["train"], sources=['features'], load_in_memory=True)
        data_valid = H5PYDataset(fname, which_sets=["valid"], sources=['features'], load_in_memory=True)
        data_test  = H5PYDataset(fname, which_sets=["test"], sources=['features'], load_in_memory=True)

        some_features = data_train.get_data(None, slice(0, 100))[0]
        assert some_features.shape[0] == 100 

        some_features = some_features.reshape([100, -1])
        x_dim = some_features.shape[1]
    else:
        raise ValueError("Unknown dataset %s" % data_name)

    return x_dim, data_train, data_valid, data_test
