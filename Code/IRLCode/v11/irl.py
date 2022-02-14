"""
Contains multiple methods of performing inverse reinforcement learning for analysis
"""

import numpy as np
from time import sleep

# Max Ent IRL ###########################################################################

# Main call - Not working yet - why?
def m_irl(world, e_features, p_initial, epochs, lr):
	n_states = world.state_space
	n_actions = world.action_space
	n_features = world.state_space
	p_transition = world.transition_prob

	# print(p_transition.shape)
	# exit()
	alpha = np.random.rand(n_states)
	for i in range(epochs):
		r  = world.features.dot(alpha) # Essentially just alpha but could have different features

		# Backwards pass
		z_s, z_a =backwards_pass(world, alpha)


		policy = local_action_probability(z_s, z_a)

		e_svf = expected_svf_from_policy(world, p_initial, policy)

		grad = e_features - world.features.T.dot(e_svf)

		alpha*=np.exp(lr*grad)

		# delta = np.max(np.abs(alpha_old - alpha))
	
	return world.features.dot(alpha)


# Expected state visitatoin

def backwards_pass(world, rewards):
	n_states, n_actions = world.state_space, world.action_space
	z_states = np.zeros((n_states))
	z_action = np.zeros((n_states, n_actions))
	p_transition = world.transition_prob
	er = np.exp(rewards)*np.eye((n_states))
	ee = np.exp(rewards)
	tmp_s = np.zeros((n_states))
	tmp_a = np.zeros((n_states, n_actions))

	for a in range(len(world.terminals)):
		end = world.terminals[a]
		z_states[end] = 1.0
		tmp_s[end] = 1.0

	# for o in range(2*n_states):

	# 	for i in range(n_states):

	# 		for j in range(n_actions):
	# 			summed = 0
	# 			for k in range(n_states):

	# 				c1 = p_transition[i, k, j]
			
	# 				assert c1 <= 1
	# 				c2 = ee[i]

	# 				c3 = z_states[k]

	# 				summed += c1*c2*c3
	# 			z_action[i,j] = summed

	# 		z_states[i] = np.sum(z_action[i,:])

			# z_states[i] = z_states[i] + 1 if i in world.terminals else z_states[i]
			# exit()
		# if o == 0:
		# 	print(z_action)
		# 	print(z_states)
		# 	exit()

	for o in range(2*n_states):

		for a in range(n_actions):
			tmp_a[:,a] = np.matmul(er, np.matmul(p_transition[:,:,a], tmp_s.T))/2
		
		tmp_s = tmp_a.sum(axis=1)

	return (tmp_s, tmp_a)

def local_action_probability(z_states, z_action):
	policy = z_action/z_states[:,None]
	return policy

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
# Linear Inverse Reinforcement Learning ################################################


# Functions for everyone ###############################################################

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