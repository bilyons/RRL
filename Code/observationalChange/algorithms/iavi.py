"""
Deep Inverse Q-Learning with Constraints. NeurIPS 2020.
Gabriel Kalweit, Maria Huegle, Moritz Werling and Joschka Boedecker
Neurorobotics Lab, University of Freiburg.
"""

import numpy as np

epsilon = 1e-6

def action_probabilities(world, trajectories):
	n_states, n_actions, _ = world.transition_prob.shape

	action_probabilities = np.zeros((n_states, n_actions))
	for t in trajectories:
		for s in t.states_actions():
			action_probabilities[s[0], s[1]] += 1
	action_probabilities[action_probabilities.sum(axis=1)==0]=1e-5
	action_probabilities /= action_probabilities.sum(axis=1).reshape(n_states, 1)
	return action_probabilities

def inverse_action_value_iteration(world, discount, trajectories, eps=1e-4):
	n_states, n_actions, _ = world.transition_prob.shape
	transition_probabilities = world.transition_prob

	# Calculate Action probabilities
	a_p = action_probabilities(world, trajectories)

	# Initialise tables
	r = np.zeros((n_states, n_actions))
	q = np.zeros((n_states, n_actions))

	# Compute reverse topological order
	T = []
	for i in reversed(range(n_states)):
		T.append([i])

	diff = np.inf
	while diff > eps:
		diff = 0
		for t in T[0:]:
			for i in t:
				X = []
				for a in range(n_actions):
					row = np.ones(n_actions)
					for oa in range(n_actions):
						if oa==a:
							continue
						row[oa] /= -(n_actions-1)
					X.append(row)
				X = np.array(X)

				y = []
				for a in range(n_actions):
					other_actions = [oa for oa in range(n_actions) if oa != a]
					sum_of_oa_logs = np.sum([np.log(a_p[i][oa] + epsilon) for oa in other_actions])
					sum_of_oa_q = np.sum([transition_probabilities[i][oa] * discount * np.max(q[np.arange(n_states)], axis=1) for oa in other_actions])
					y.append(np.log(a_p[i][a] + epsilon)-(1/(n_actions-1))*sum_of_oa_logs+(1/(n_actions-1))*sum_of_oa_q-np.sum(transition_probabilities[i][a] * discount * np.max(q[np.arange(n_states)], axis=1)))
				y = np.array(y)

				# Find least-squares solution.
				x = np.linalg.lstsq(X, y, rcond=None)[0]
					
				for a in range(n_actions):
					diff = max(np.abs(r[i, a]-x[a]), diff)

				# compute new r and Q-values.
				r[i] = x
				for a in range(n_actions):
					q[i, a] = r[i, a] + np.sum(transition_probabilities[i][a] * discount * np.max(q[np.arange(n_states)], axis=1))
	
	# calculate Boltzmann distribution.
	boltzman_distribution = []
	for s in range(n_states):
		boltzman_distribution.append([])
		for a in range(n_actions):
			boltzman_distribution[-1].append(np.exp(q[s][a]))
	boltzman_distribution = np.array(boltzman_distribution)
	boltzman_distribution /= np.sum(boltzman_distribution, axis=1).reshape(-1, 1)
	return q, r, boltzman_distribution