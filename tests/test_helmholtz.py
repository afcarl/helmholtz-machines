
import unittest 

from helmholtz.prob_layers import ProbabilisticTopLayer, ProbabilisticLayer
from helmholtz import *


def test_create_layers():
    p_layers, q_layers = create_layers("20,10", 50)

    assert len(p_layers) == 3
    assert len(q_layers) == 2

    assert isinstance(p_layers[-1], ProbabilisticTopLayer)
    for layer in p_layers[:-1]:
        assert isinstance(layer, ProbabilisticLayer)

    for layer in q_layers:
        assert isinstance(layer, ProbabilisticLayer)

def test_flatten_values():
    pass

def test_unflatten_values():
    pass

