"""
Contains multiple methods of performing inverse reinforcement learning for analysis
"""

import numpy as np
from time import sleep

# Max Ent IRL ###########################################################################

# Main call
def m_irl(world, e_features, p_initial, epochs, lr, alpha):

	n_states = world.state_space
	n_actions = world.action_space
	_, n_features = world.features.shape
	p_transition = world.transition_prob

	# alpha = np.ones(n_states)
	alpha = np.random.rand(n_states)
	# Exponential gradient descent
	for i in range(epochs):
		r = world.features.dot(alpha)

		e_svf = expected_svf(world, p_initial, r)
		grad = e_features - world.features.T.dot(e_svf)

		# print("expected")
		# print(e_features.reshape((5,5)))
		# print("believed")
		# print(world.features.T.dot(e_svf))
		# sleep(0.1)
		alpha *= np.exp(lr*grad)

	return world.features.dot(alpha)

# Expected state visitatoin

def expected_svf(world, p_initial, reward):
	p_action = local_action_probabilities(world, reward)
	return expected_svf_from_policy(world, p_initial, p_action)

def local_action_probabilities(world, reward):
	n_states, n_actions = world.state_space, world.action_space

	er = np.exp(reward)
	# print(er)
	er = er/np.max(er)
	p = world.p_transition
	# Initialise at terminal state
	zs = np.zeros(n_states)
	for a in range(len(world.terminals)):
		end = world.terminals[a]
		zs[end] = 1.0

	for _ in range(2*n_states):
		za = np.array([er*p[a].dot(zs) for a in range(n_actions)]/np.array([2])).T
		zs = za.sum(axis=1)


	# print(za)
	# print(zs[:, None])
	# print(za/zs[:, None])
	# sleep(0.1)
	return za/zs[:, None]

def expected_svf_from_policy(world, p_initial, p_action, eps = 1e-5):
	n_states, n_actions = world.state_space, world.action_space
	p_transition = np.copy(world.transition_prob)
	for a in range(len(world.terminals)):
		end = world.terminals[a]
		p_transition[end,:,:] = 0.0
	p_transition = [np.array(p_transition[:,:,a]) for a in range(n_actions)]

	# forward computation of state expectations
	d = np.zeros(n_states)

	delta = np.inf

	while delta > eps:
		# print([p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)])
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_
		# print(d)

	return d

# Functions for everyone ###############################################################

def feature_expectations_updatable(world, trajectories, feat_count, ep):
	# Updatable feature expectation, takes in previous counts and a batch of trajectories >= 1
	# And returns new expectation and count
	n_states, n_features = world.features.shape

	fe = feat_count

	for t in trajectories:
		for s in t.states():
			fe += world.features[s,:]
	return fe/(ep), fe

def inital_probabilities_updatable(world, trajectories, init_count, ep):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	p = init_count

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p/ (ep), p

def feature_expectations_addable(world, trajectories, feat_count, number_adding):
	n_states, n_features = world.features.shape

	fe = feat_count

	for t in trajectories[-number_adding:]:
		for s in t.states():
			fe += world.features[s,:]
	return fe

def inital_probabilities_addable(world, trajectories, init_count, number_adding):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	p = init_count

	for t in trajectories[-number_adding:]:
		p[t.transitions()[0][0]] += 1.0
	return p