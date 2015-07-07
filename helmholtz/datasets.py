
from __future__ import division


local_datasets = ["adult", "dna", "web", "nips", "mushrooms", "ocr_letters", "connect4", "rcv1"]
supported_datasets = local_datasets + ['bmnist', 'bars', 'silhouettes']

# 'tfd' is missing but needs normalization 

def get_data(data_name):
    if data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        x_dim = 28*28 

        data_train = BinarizedMNIST(which_sets=['train'], sources=['features'])
        data_valid = BinarizedMNIST(which_sets=['valid'], sources=['features'])
        data_test  = BinarizedMNIST(which_sets=['test'], sources=['features'])
    elif data_name == 'silhouettes':
        from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes

        size = 28
        x_dim = size*size

        data_train = CalTech101Silhouettes(which_sets=['train'], size=size, sources=['features'])
        data_valid = CalTech101Silhouettes(which_sets=['valid'], size=size, sources=['features'])
        data_test  = CalTech101Silhouettes(which_sets=['test'], size=size, sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase

        size = 48
        x_dim = size*size

        data_train = TorontoFaceDatabase(which_sets=['unlabeled'], size=size, sources=['features'])
        data_valid = TorontoFaceDatabase(which_sets=['valid'], size=size, sources=['features'])
        data_test  = TorontoFaceDatabase(which_sets=['test'], size=size, sources=['features'])
    elif data_name == 'bars':
        from bars_data import Bars

        width = 4
        x_dim = width*width

        data_train = Bars(num_examples=5000, width=width, sources=['features'])
        data_valid = Bars(num_examples=5000, width=width, sources=['features'])
        data_test  = Bars(num_examples=5000, width=width, sources=['features'])
    elif data_name in local_datasets:
        from fuel.datasets.hdf5 import H5PYDataset

        fname = "data/"+data_name+".hdf5"
        
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
