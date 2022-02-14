"""
Generic stochastic gradient-ascent based optimizers.

Due to the MaxEnt IRL objective of maximizing the log-likelihood instead of
minimizing a loss function, all optimizers in this module are actually
stochastic gradient-ascent based instead of the more typical descent.
"""

import numpy as np


class Optimizer:
    """
    Optimizer base-class.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
    """
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments are optimizer-specific.
        """
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        """
        Create a new wrapper for this optimizer which normalizes the
        gradient before each step.

        Returns:
            An Optimizer instance wrapping this Optimizer, normalizing the
            gradient before each step.

        See also:
            `class NormalizeGrad`
        """
        return NormalizeGrad(self, ord)

class ExpSga(Optimizer):
    """
    Exponentiated stochastic gradient ascent.

    The implementation follows Algorithm 10.5 from B. Ziebart's thesis
    (2010) and is slightly adapted from the original algorithm provided by
    Kivinen and Warmuth (1997).

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`
            taking the step number as parameter and returning a learning
            rate as result.
            See also `linear_decay`, `power_decay` and `exponential_decay`.
        normalize: A boolean specifying if the the parameters should be
            normalized after each step, as done in the original algorithm by
            Kivinen and Warmuth (1997).

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
        lr: The learning-rate as specified in the __init__ function.
        k: The number of steps run since the last reset.
    """
    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad)

        if self.normalize:
            self.parameters /= self.parameters.sum()

def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    """
    Linear learning-rate decay.

    Creates a function `(k: Integer) -> learning_rate: Float` returning the
    learning-rate in dependence on the current number of iterations. The
    returned function can be expressed as

        learning_rate(k) = lr0 / (1.0 + decay_rate * floor(k / decay_steps))

    Args:
        lr0: The initial learning-rate.
        decay_rate: The decay factor.
        decay_steps: An integer number of steps that can be used to
            staircase the learning-rate.

    Returns:
        The function giving the current learning-rate in dependence of the
        current iteration as specified above.
    """
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr