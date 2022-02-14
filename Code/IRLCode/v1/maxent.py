import numpy as np

def irl(world, features, terminals, trajectories, optim, init, eps=1e-4, eps_esvf=1e-5):

	n_states = world.state_space
	n_actions = world.action_space
	_, n_features = features.shape
	p_transition = world.transition_prob

	# Static properties from trajectories
	e_features = feature_expectation_from_trajectories(features, trajectories)
	p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

	print(p_initial.reshape(5,5))
	# print(np.reshape(e_features, (5,5)))
	# print(np.reshape(p_initial, (5,5)))
	# exit()
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
	return features.dot(theta)

def feature_expectation_from_trajectories(features, trajectories):
	n_states, n_features = features.shape

	fe = np.zeros(n_features)

	for t in trajectories:
		for s in t.states():
			fe += features[s,:]
	return fe/len(trajectories)

def initial_probabilities_from_trajectories(n_states, trajectories):
	# Should this be from the sample or provided as initial knowledge?
	# i.e. whilst there may be some sampling discrepency, the probability should in fact
	# be
	p = np.zeros(n_states)

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p / len(trajectories)

def expected_svf_from_policy(p_transition, p_initial, terminals, p_action, eps=1e-5):
	n_states, _, n_actions = p_transition.shape

	# Can't leave terminal states
	p_transition = np.copy(p_transition)
	for a in range(len(terminals)):
		end = terminals[a]
		p_transition[end,:,:] = 0.0

	p_transition = [np.array(p_transition[:,:,a]) for a in range(n_actions)]

	# forward computation of state expectations
	d = np.zeros(n_states)

	delta = np.inf

	while delta > eps:
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_

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

	er = np.exp(reward)
	p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
	# print(p)
	# exit()

	# initialize at terminal states
	zs = np.zeros(n_states)
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