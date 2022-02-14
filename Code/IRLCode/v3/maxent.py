import numpy as np

def irl(world, features, terminals, trajectory, optim, init, feat_count, init_count, ep_number, eps=1e-4, eps_esvf=1e-5):

	n_states = world.state_space
	n_actions = world.action_space
	_, n_features = features.shape
	p_transition = world.transition_prob

	e_features, feat_count = feature_expectation_updatable(features, feat_count, trajectory, ep_number)
	p_initial, init_count = initial_probabilities_updatable(n_states, init_count, trajectory, ep_number)

	# p_initial = initial_probs_fair(n_states, terminals)
	# print(p_initial.reshape(5,5))
	# basic gradient descent
	theta = init(n_features)
	delta = np.inf
	optim.reset(theta)
	while delta > eps:
		theta_old = theta.copy()

		# compute per state reward
		reward = features.dot(theta)

		# compute gradient
		e_svf = compute_expected_svf(p_transition, p_initial, terminals, reward, eps_esvf)
		grad = e_features - features.T.dot(e_svf)

		# perform optimization step and compute delta for convergence
		optim.step(grad)
		delta = np.max(np.abs(theta_old - theta))

	return features.dot(theta), feat_count, init_count

def irl_causal(world, features, terminals, trajectory, optim, init, feat_count, init_count, ep_number, discount,
			   eps=1e-4, eps_svf=1e-5, eps_lap=1e-5):
	"""
	Compute the reward signal given the demonstration trajectories using the
	maximum causal entropy inverse reinforcement learning algorithm proposed
	Ziebart's thesis (2010).

	Args:
		p_transition: The transition probabilities of the MDP as table
			`[from: Integer, to: Integer, action: Integer] -> probability: Float`
			specifying the probability of a transition from state `from` to
			state `to` via action `action` to succeed.
		features: The feature-matrix (e.g. as numpy array), mapping states
			to features, i.e. a matrix of shape (n_states x n_features).
		terminal: Either the terminal reward function or a collection of
			terminal states. Iff `len(terminal)` is equal to the number of
			states, it is assumed to contain the terminal reward function
			(phi) as specified in Ziebart's thesis. Otherwise `terminal` is
			assumed to be a collection of terminal states from which the
			terminal reward function will be derived.
		trajectories: A list of `Trajectory` instances representing the
			expert demonstrations.
		optim: The `Optimizer` instance to use for gradient-based
			optimization.
		init: The `Initializer` to use for initialization of the reward
			function parameters.
		discount: A discounting factor for the log partition functions as
			Float.
		eps: The threshold to be used as convergence criterion for the
			reward parameters. Convergence is assumed if all changes in the
			scalar parameters are less than the threshold in a single
			iteration.
		eps_lap: The threshold to be used as convergence criterion for the
			state partition function. Convergence is assumed if the state
			partition function changes less than the threshold on all states
			in a single iteration.
		eps_svf: The threshold to be used as convergence criterion for the
			expected state-visitation frequency. Convergence is assumed if
			the expected state visitation frequency changes less than the
			threshold on all states in a single iteration.
	"""
	n_states = world.state_space
	n_actions = world.action_space
	_, n_features = features.shape
	p_transition = world.transition_prob

	# compute static properties from trajectories
	e_features = feat_count/ (ep_number+1)
	p_initial = init_count / (ep_number+1)
	# basic gradient descent
	theta = init(n_features)
	delta = np.inf

	optim.reset(theta)
	while delta > eps:
		theta_old = theta.copy()

		# compute per-state reward
		reward = features.dot(theta)

		# compute the gradient
		e_svf = compute_expected_causal_svf(p_transition, p_initial, terminals, reward, discount,
											eps_lap, eps_svf)

		grad = e_features - features.T.dot(e_svf)

		# perform optimization step and compute delta for convergence
		optim.step(grad)
		delta = np.max(np.abs(theta_old - theta))

	# re-compute per-state reward and return
	return features.dot(theta)

def feature_expectation_updatable(features, old_count, trajectory, ep_number):
	n_states, n_features = features.shape

	fe = old_count

	for s in trajectory.states():
		fe += features[s,:]

	return fe/(ep_number+1), fe

def initial_probabilities_updatable(n_states, old_count, trajectory, ep_number):
	p = old_count

	p[trajectory.transitions()[0][0]] += 1.0
	return p / (ep_number+1), p

def initial_probs_fair(n_states, terminals):
	p=np.zeros(n_states)
	for t in range(n_states):
		if t not in terminals:
			p[t] = 1/(n_states-len(terminals))
	return p

def expected_svf_from_policy(p_transition, p_initial, terminals, p_action, eps=1e-5):
	n_states, _, n_actions = p_transition.shape

	# Can't leave terminal states
	p_transition = np.copy(p_transition)
	# print(p_transition)
	# exit()
	for a in range(len(terminals)):
		end = terminals[a]
		p_transition[end,:,:] = 0.0

	p_transition = [np.array(p_transition[:,:,a]) for a in range(n_actions)]

	# forward computation of state expectations
	d = np.zeros(n_states)

	delta = np.inf

	while delta > eps:
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		# print([p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)])
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_
	# exit()
	return d

def local_action_probabilities(p_transition, terminals, reward):
	"""
	Compute the local action probabilities (policy) required for the edge
	frequency calculation for maximum entropy reinfocement learning.
	This is the backward pass of Algorithm 1 of the Maximum Entropy IRL
	paper by Ziebart et al. (2008).
	Args:
		p_transition: The transition probabilities of the MDP as table
			`[from: Integer, to: Integer, action: Integer] -> probability: Float`
			specifying the probability of a transition from state `from` to
			state `to` via action `action` to succeed.
		terminal: A set/list of terminal states.
		reward: The reward signal per state as table
			`[state: Integer] -> reward: Float`.
	Returns:
		The local action probabilities (policy) as map
		`[state: Integer, action: Integer] -> probability: Float`
	"""
	n_states, _, n_actions = p_transition.shape

	# print(n_states)

	er = np.exp(reward)
	p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

	print(p)
	exit()

	# initialize at terminal states
	zs = np.zeros(n_states)
	# print(p)
	# print(zs)
	for a in range(len(terminals)):
		end = terminals[a]
		zs[end] = 1.0

	# perform backward pass
	# This does not converge, instead we iterate a fixed number of steps. The
	# number of steps is chosen to reflect the maximum steps required to
	# guarantee propagation from any state to any other state and back in an
	# arbitrary MDP defined by p_transition.
	for _ in range(2 * n_states):
		za = np.array([er * p[a].dot(zs) for a in range(n_actions)]).T
		zs = za.sum(axis=1)

	# compute local action probabilities
	return za / zs[:, None]


def compute_expected_svf(p_transition, p_initial, terminals, reward, eps=1e-5):
	"""
	Compute the expected state visitation frequency for maximum entropy IRL.
	This is an implementation of Algorithm 1 of the Maximum Entropy IRL
	paper by Ziebart et al. (2008).
	This function combines the backward pass implemented in
	`local_action_probabilities` with the forward pass implemented in
	`expected_svf_from_policy`.
	Args:
		p_transition: The transition probabilities of the MDP as table
			`[from: Integer, to: Integer, action: Integer] -> probability: Float`
			specifying the probability of a transition from state `from` to
			state `to` via action `action` to succeed.
		p_initial: The probability of a state being an initial state as map
			`[state: Integer] -> probability: Float`.
		terminal: A list of terminal states.
		reward: The reward signal per state as table
			`[state: Integer] -> reward: Float`.
		eps: The threshold to be used as convergence criterion for the
			expected state-visitation frequency. Convergence is assumed if
			the expected state visitation frequency changes less than the
			threshold on all states in a single iteration.
	Returns:
		The expected state visitation frequencies as map
		`[state: Integer] -> svf: Float`.
	"""
	p_action = local_action_probabilities(p_transition, terminals, reward)
	return expected_svf_from_policy(p_transition, p_initial, terminals, p_action, eps)

def softmax(x1, x2):
	"""
	Computes a soft maximum of both arguments.

	In case `x1` and `x2` are arrays, computes the element-wise softmax.

	Args:
		x1: Scalar or ndarray.
		x2: Scalar or ndarray.

	Returns:
		The soft maximum of the given arguments, either scalar or ndarray,
		depending on the input.
	"""
	x_max = np.maximum(x1, x2)
	x_min = np.minimum(x1, x2)
	return x_max + np.log(1.0 + np.exp(x_min - x_max))


def local_causal_action_probabilities(p_transition, terminals, reward, discount, eps=1e-5):
	"""
	Compute the local action probabilities (policy) required for the edge
	frequency calculation for maximum causal entropy reinfocement learning.

	This is Algorithm 9.1 from Ziebart's thesis (2010) combined with
	discounting for convergence reasons as proposed in the same thesis.

	Args:
		p_transition: The transition probabilities of the MDP as table
			`[from: Integer, to: Integer, action: Integer] -> probability: Float`
			specifying the probability of a transition from state `from` to
			state `to` via action `action` to succeed.
		terminal: Either the terminal reward function or a collection of
			terminal states. Iff `len(terminal)` is equal to the number of
			states, it is assumed to contain the terminal reward function
			(phi) as specified in Ziebart's thesis. Otherwise `terminal` is
			assumed to be a collection of terminal states from which the
			terminal reward function will be derived.
		reward: The reward signal per state as table
			`[state: Integer] -> reward: Float`.
		discount: A discounting factor as Float.
		eps: The threshold to be used as convergence criterion for the state
			partition function. Convergence is assumed if the state
			partition function changes less than the threshold on all states
			in a single iteration.

	Returns:
		The local action probabilities (policy) as map
		`[state: Integer, action: Integer] -> probability: Float`
	"""
	n_states, _, n_actions = p_transition.shape

	# set up terminal reward function
	if len(terminals) == n_states:
		reward_terminal = np.array(terminal, dtype=np.float)
	else:
		reward_terminal = -np.inf * np.ones(n_states)
		for t in terminals:
			reward_terminal[t] = 0.0

	# set up transition probability matrices
	p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

	# compute state log partition V and state-action log partition Q
	v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

	delta = np.inf
	while delta > eps:
		v_old = v

		q = np.array([reward + discount * p[a].dot(v_old) for a in range(n_actions)]).T

		v = reward_terminal
		for a in range(n_actions):
			v = softmax(v, q[:, a])

		# for some reason numpy chooses an array of objects after reduction, force floats here
		v = np.array(v, dtype=np.float)

		delta = np.max(np.abs(v - v_old))

	# compute and return policy
	return np.exp(q - v[:, None])


def compute_expected_causal_svf(p_transition, p_initial, terminals, rewards, discount,
								eps_lap=1e-5, eps_svf=1e-5):
	"""
	Compute the expected state visitation frequency for maximum causal
	entropy IRL.

	This is a combination of Algorithm 9.1 and 9.3 of Ziebart's thesis
	(2010). See `local_causal_action_probabilities` and
	`expected_svf_from_policy` for more details.

	Args:
		p_transition: The transition probabilities of the MDP as table
			`[from: Integer, to: Integer, action: Integer] -> probability: Float`
			specifying the probability of a transition from state `from` to
			state `to` via action `action` to succeed.
		p_initial: The probability of a state being an initial state as map
			`[state: Integer] -> probability: Float`.
		terminal: Either the terminal reward function or a collection of
			terminal states. Iff `len(terminal)` is equal to the number of
			states, it is assumed to contain the terminal reward function
			(phi) as specified in Ziebart's thesis. Otherwise `terminal` is
			assumed to be a collection of terminal states from which the
			terminal reward function will be derived.
		reward: The reward signal per state as table
			`[state: Integer] -> reward: Float`.
		discount: A discounting factor as Float.
		eps_lap: The threshold to be used as convergence criterion for the
			state partition function. Convergence is assumed if the state
			partition function changes less than the threshold on all states
			in a single iteration.
		eps_svf: The threshold to be used as convergence criterion for the
			expected state-visitation frequency. Convergence is assumed if
			the expected state visitation frequency changes less than the
			threshold on all states in a single iteration.
	"""
	p_action = local_causal_action_probabilities(p_transition, terminals, rewards, discount, eps_lap)
	return expected_svf_from_policy(p_transition, p_initial, terminals, p_action, eps_svf)