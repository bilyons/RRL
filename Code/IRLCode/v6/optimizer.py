import matplotlib 
matplotlib.use('Agg') 
import numpy as np

"""
Optimizers for maximising log likelihood.
All performing stochastic gradient ascent.
"""

class Initializer:
	"""
	Base-class for an Initializer, specifying a strategy for parameter
	initialization.
	"""
	def __init__(self):
		pass

	def initialize(self, shape):
		"""
		Create an initial set of parameters.
		Args:
			shape: The shape of the parameters.
		Returns:
			An initial set of parameters of the given shape, adhering to the
			initialization-strategy described by this Initializer.
		"""
		raise NotImplementedError

	def __call__(self, shape):
		"""
		Create an initial set of parameters.
		Note:
			This function simply calls `self.initialize(shape)`.
		Args:
			shape: The shape of the parameters.
		Returns:
			An initial set of parameters of the given shape, adhering to the
			initialization-strategy described by this Initializer.
		"""
		return self.initialize(shape)


class Uniform(Initializer):
	"""
	An Initializer, initializing parameters according to a specified uniform
	distribution.
	Args:
		low: The minimum value of the distribution.
		high: The maximum value of the distribution
	Attributes:
		low: The minimum value of the distribution.
		high: The maximum value of the distribution
	"""
	def __init__(self, low=0.0, high=1.0):
		super().__init__()
		self.low = low
		self.high = high

	def initialize(self, shape):
		"""
		Create an initial set of uniformly random distributed parameters.
		The parameters of the distribution can be specified in the
		constructor.
		Args:
			shape: The shape of the parameters.
		Returns:
			An set of initial uniformly distributed parameters of the given
			shape.
		"""
		return np.random.uniform(size=shape, low=self.low, high=self.high)


class Constant(Initializer):
	"""
	An Initializer, initializing parameters to a constant value.
	Args:
		value: Either a scalar value or a function in dependence on the
			shape of the parameters, returning a scalar value for
			initialization.
	"""
	def __init__(self, value=1.0):
		super().__init__()
		self.value = value

	def initialize(self, shape):
		"""
		Create set of parameters with initial fixed value.
		The scalar value used for initialization can be specified in the
		constructor.
		Args:
			shape: The shape of the parameters.
		Returns:
			An set of constant-valued parameters of the given shape.
		"""
		if callable(self.value):
			return np.ones(shape) * self.value(shape)
		else:
			return np.ones(shape) * self.value


class Optimizer:
	"""
	Optimizer base-class.

	Note:
	Before use of any optimizer, its reset function must be called.
	"""

	def __init__(self):
		self.parameters = None

	def reset(self, parameters):
		# Reset the optimizer
		self.parameters = parameters

	def step(self, grad, *args, **kwargs):
		raise NotImplementedError

	def normalize_grad(self, ord=None):
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