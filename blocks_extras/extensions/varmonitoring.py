
import logging

from collections import OrderedDict

import theano
from theano import tensor
from theano.ifelse import ifelse

from blocks.algorithms import DifferentiableCostMinimizer
from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import AggregationBuffer, DatasetEvaluator
from blocks.monitoring.aggregation import AggregationScheme, Aggregator, mean, MonitoredQuantity
from blocks.utils import dict_subset, shared_like


logger = logging.getLogger()

class VarianceMonitoring(SimpleExtension, MonitoringExtension):
    """Monitors Theano variables and monitored-quantities on a data stream.

    By default monitoring is done before the first and after every epoch.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variables to monitor. The variable names are used as record
        names in the logs.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningful way. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to :func:`~theano.scan` which
        might have returned shared variable updates.
    data_stream : instance of :class:`.DataStream`
        The data stream to monitor on. A data epoch is requested
        each time monitoring is done.

    """
    def __init__(self, variables, data_stream, repeats=100, updates=None, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        super(VarianceMonitoring, self).__init__(**kwargs)
        self._evaluator = VarianceEvaluator(variables, repeats, updates)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Variance monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Variance monitoring on auxiliary data finished")

#----------------------------------------------------------------------------


class StandardDeviation(AggregationScheme):
    """Aggregation scheme which computes the Variance
    """
    def __init__(self, variable, denominator):
        self.variable = tensor.as_tensor(variable)
        self.denominator = tensor.as_tensor(denominator)

    def get_aggregator(self):
        initialized = shared_like(0.)                   # initialize to correct shape?
        variable_acc = shared_like(self.variable)       # aggregate var
        variable_sq_acc = shared_like(self.variable**2) # aggregate var**2
        denominator_acc = shared_like(self.denominator) # aggregate N

        variable_zeros = tensor.as_tensor(self.variable).zeros_like()
        denominator_zeros = tensor.as_tensor(self.denominator).zeros_like()

        variable_update = self.variable + ifelse(initialized,
                                                    variable_acc,
                                                    variable_zeros)
        variable_sq_update = self.variable**2 + ifelse(initialized,
                                                    variable_sq_acc,
                                                    variable_zeros)
        denominator_update = self.denominator + ifelse(initialized,
                                                    denominator_acc,
                                                    denominator_zeros)

        initialization_updates = [(initialized, 0.)]
        #initialization_updates = [(variable_acc, variable_zeros),
        #                          (variable_sq_acc, variable_zeros),
        #                          (denominator_acc, denominator_zeros),
        #                          (initialized, 0.)]

        #accumulation_updates = []
        accumulation_updates = [(variable_acc, variable_update),
                                (variable_sq_acc, variable_sq_update),
                                (denominator_acc, denominator_update),
                                (initialized, 1.)]

        readout = tensor.sqrt(
                    (denominator_acc * variable_sq_acc - variable_acc ** 2)
                        / denominator_acc / (denominator_acc-1)
                   )

        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable=readout)

        return aggregator

def std(variable, axis=None):
    """ Compute the aggregated standard deviation """
    agg = 0*variable
    agg.tag.aggregation_scheme = StandardDeviation(variable, 1.0)
    agg.name = "{}_std".format(variable.name)
    return agg

#----------------------------------------------------------------------------


class VarianceEvaluator(object):
    """A DatasetEvaluator evaluates many Theano variables or other quantities.

    The DatasetEvaluator provides a do-it-all method, :meth:`evaluate`,
    which computes values of ``variables`` on a dataset.

    Alternatively, methods :meth:`initialize_aggregators`,
    :meth:`process_batch`, :meth:`get_aggregated_values` can be used with a
    custom loop over data.

    The values computed on subsets of the given dataset are aggregated
    using the :class:`AggregationScheme`s provided in the
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variable names are used as record names in the logs. Hence, all
        the names must be different.

        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningfullway. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to:function:`~theano.scan` which
        might have returned shared variable updates.

    """
    def __init__(self, variables, repeats, updates=None):
        variable_names = [v.name for v in variables]
        if len(set(variable_names)) < len(variables):
            raise ValueError("variables should have different names")
        theano_variables = []
        for variable in variables:
            if isinstance(variable, MonitoredQuantity):
                raise ValueError("VarianceEvaluator does not support MonitoredQuantity")
            var_mean = mean(variable)
            var_mean.name = "{}_mean_abs".format(variable.name)
            theano_variables.append(var_mean)
            theano_variables.append(std(variable))
        self.theano_variables = theano_variables
        self.repeats = repeats
        self.theano_buffer = AggregationBuffer(theano_variables)
        self.updates = updates
        self._compile()

    def _compile(self):
        """Compiles Theano functions. """
        inputs = []
        outputs = []
        updates = OrderedDict()
        if self.updates is not None:
            updates.update(self.updates)
        if self.theano_buffer.accumulation_updates:
            updates.update(self.theano_buffer.accumulation_updates)
            inputs += self.theano_buffer.inputs

        if inputs == []:
            self._accumulate_fun = None
        else:
            self.unique_inputs = list(set(inputs))
            self._accumulate_fun = theano.function(self.unique_inputs,
                                                   outputs,
                                                   updates=updates)

    def reset_statistics(self):
        self.statistics_merge_count = 0
        self.statistics = {}

    def merge_statistics(self, statistics):
        self.statistics_merge_count += 1
        for k, v in statistics.items():
            if k not in self.statistics:
                self.statistics[k] = abs(v).mean()
            else:
                self.statistics[k] += abs(v).mean()

    def get_aggregated_values(self):
        return {k: v / self.statistics_merge_count for k, v in self.statistics.items()}

    def process_batch(self, batch):
        if self._accumulate_fun is None:
            return

        try:
            input_names = [v.name for v in self.unique_inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))

        self.theano_buffer.initialize_aggregators()
        for _ in xrange(self.repeats):
            self._accumulate_fun(**batch)
        self.merge_statistics(self.theano_buffer.get_aggregated_values())

    def evaluate(self, data_stream):
        """Compute the variables over a data stream.

        Parameters
        ----------
        data_stream : instance of :class:`.DataStream`
            The data stream. Only the first epoch of data is used.

        Returns
        -------
        A mapping from record names to the values computed on the provided
        dataset.

        """
        if self._accumulate_fun is None:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        self.reset_statistics()
        for batch in data_stream.get_epoch_iterator(as_dict=True):
            self.process_batch(batch)

        return self.get_aggregated_values()
