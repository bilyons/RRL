"""
Find the value function associated with a given policy.

Billy Lyons, 2021
billy.lyons@ed.ac.uk

Adapted from Matthew Alger: https://github.com/MatthewJA/Inverse-Reinforcement-Learning

Based on Sutton and Barto
"""

import numpy as np
import time

def value_iteration(world, p_transition, rewards, gamma, error= 0.001):

	"""
	We assume the policy is static but the reflexive agent performs
	in suboptimal manners to make IRL rewards appear closer to the truth.

	As such we don't have to recency weight and can assume all trajectories are
	reasonable.

	inputs:
		p_transition: transition probabilities of the MPD. This is an |S|x|S|x|A| array
			where p_transition[s, s', a] is the probability of going to state s' from s
			under action a
		rewards: |S|x1 vector of rewards.
		gamma: discount factor
		error: threshold at which we consider it to be solved

	returns:
		values: |S|x1 vector of value function
		policy: |S|x|A| array of probabilities of choosing an action in a given state
	"""

	n_states, _, n_actions = np.shape(p_transition)
	v = np.zeros(n_states)
	i=0
	diff = np.inf

	# policy = np.zeros((n_states, n_actions))

	# p = [np.matrix(p_transition[:, :, a]) for a in range(n_actions)]
	# # print(v.reshape((11,11)))
	# delta = np.inf
	# v = np.zeros(n_states)
	# start = time.time()
	# while delta > error:      # iterate until convergence
	# 	v_old = v

	# 	# compute state-action values (note: we actually have Q[a, s] here)
	# 	# q = gamma * np.array([p[a] @ v for a in range(n_actions)])

	# 	# # compute state values
	# 	# v = rewards + np.average(q, axis=0)[0]

	# 	# # compute state-action values (note: we actually have Q[a, s] here)
	# 	q = gamma * np.array([p[a] @ v for a in range(n_actions)])

	# 	# compute state values
	# 	v = rewards + np.max(q, axis=0)[0]
	# 	# print(v)

	# 	# compute maximum delta
	# 	delta = np.max(np.abs(v_old - v))
	# end = time.time()

	# print(v.reshape((11,11)))

	# v = np.zeros(n_states)

	diff = float("inf")
	while diff > 0.001:
		diff = 0
		for s in range(n_states):
			max_v = float("-inf")
			for a in range(n_actions):
				tp = p_transition[s, :, a]
				max_v = max(max_v, np.dot(tp, rewards + gamma*v))

			new_diff = abs(v[s] - max_v)
			if new_diff > diff:
				diff = new_diff
			v[s] = max_v	

	# # Get Q using equation 9.2 from Ziebart's thesis.
	# Q = np.zeros((n_states, n_actions))
	# for i in range(n_states):
	# 	for j in range(n_actions):
	# 		p = p_transition[i, :, j]
	# 		# print(p)
	# 		# exit()
	# 		Q[i, j] = p.dot(rewards + gamma*v)
	Q = np.zeros((n_states, n_actions))	
	for a in range(n_actions):
		Q[:,a] = np.dot(p_transition[:,:,a], (rewards+gamma*v))
	# print(pol-Q)
	Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
	Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
	# print(rewards+gamma*v)
	return v, Q

