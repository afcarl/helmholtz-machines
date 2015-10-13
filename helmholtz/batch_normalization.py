
from __future__ import division, print_function 

import logging

from collections import OrderedDict
from picklable_itertools.extras import equizip
from theano import tensor
from toolz import interleave

from blocks.bricks import Brick, Feedforward, Random, Initializable, Sequence, Linear
from blocks.bricks.base import application, lazy
from blocks.initialization import IsotropicGaussian
from blocks.roles import add_role, PARAMETER
from blocks.utils import pack, shared_floatx_zeros

logger = logging.getLogger(__name__)


class BatchNormalization(Initializable, Random):
    """A batch normalization layer.

    Parameters
    ----------
    dim : int
        Number of units to be batch normalized.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, beta_init, gamma_init, **kwargs):
        """
        """
        super(BatchNormalization, self).__init__(**kwargs)
        
        self.dim = dim
        self.beta_init = beta_init
        self.gamma_init = gamma_init

    def _initialize(self):
        self.beta  = shared_floatx_zeros( (self.dim,), name='beta')
        self.gamma = shared_floatx_zeros( (self.dim,), name='gamma')

        add_role(self.beta, PARAMETER)
        add_role(self.gamma, PARAMETER)

        self.parameters = [self.gamma, self.beta]

        self.beta_init.initialize(self.beta, self.rng)
        self.gamma_init.initialize(self.gamma, self.rng)

    @application(inputs=['x'], outputs=['x_hat'])
    def apply(self, x):
        eps = 1e-5

        mu = tensor.mean(x, axis=0)                  # shape: dim
        var = tensor.mean( (x-mu)**2, axis=0)        # shape: dim

        x_hat = (x - mu) / tensor.sqrt(var + eps)
        return self.gamma*x_hat + self.beta

        
class BatchNormalizedMLP(Sequence, Initializable, Feedforward):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.
    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, bn_init=None, **kwargs):
        if bn_init is None:
            bn_init = IsotropicGaussian(0.1)

        self.bn_init = bn_init
        self.activations = activations
        self.batch_normnalizations = [BatchNormalization(name='bn_{}'.format(i), 
                                                         beta_init=bn_init, gamma_init=bn_init)
                                       for i in range(len(activations))]
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]

        # Interleave the transformations and activations
        application_methods = []
        for entity in interleave([self.linear_transformations, self.batch_normnalizations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(BatchNormalizedMLP, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError

        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias

        for output_dim, bn in \
                equizip(self.dims[1:],
                        self.batch_normnalizations):
            bn.dim = output_dim
