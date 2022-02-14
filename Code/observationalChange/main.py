import sys, os
import numpy as np
# np.set_printoptions(suppress=True, precision=5)
from mdp.gridworld import *
from mdp.value_iteration import *
from mdp.trajectory import *
from algorithms import maxent as M
from algorithms import iavi as I
from agents import qlearning as Q
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kendalltau


def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
	vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

def main():

	# Create gridworld
	gw = StochasticGridWorld(5,0.0)

	###################################
	# Agent 1: True reward for expert #
	###################################

	# Agent 1 will not observe 2, and will instead do their own work.
	# It will know of a more valuable point at training that agent 1
	true_reward = np.zeros(gw.n_states)
	true_reward[24] = 5.0
	true_reward[4] = 2.5
	true_reward[20] = 2.0

	#######################################
	# Agent 2: Reward for observing agent #
	#######################################

	# Agent 2 will train on a different reward function and then
	# by observing unusual behaviour from 2, investigate the state
	reward = np.zeros(gw.n_states)
	reward[24] = 5.0
	reward[20] = 1.0

	# Terminal and initial states
	terminal = [24]
	initial = 0

	q_array = np.ones((gw.n_states, gw.n_actions))

	# Train
	eps = 1e-4
	delta = np.inf
	while delta>eps:
		copy = q_array.copy()
		state = initial
		t=0
		while state not in terminal:
			action = Q.action_selection(q_array, state, 2)
			next_s = range(gw.n_states)
			next_p = gw.transition_prob[state, action,:]
			next_state = np.random.choice(next_s, p=next_p)

			# print(q_array, state, action, next_state, reward[state])
			q_array = Q.q_update(q_array, state, action, next_state, reward[next_state]-1, 0.9, 0.1)

			state = next_state
			t+=1

		delta = np.max(np.abs(copy-q_array))
		# print(delta)

	# print(q_array)

	for state in range(gw.n_states):
		exponentiated = np.exp(q_array[state,:]/0.5)
		prob = exponentiated/np.sum(exponentiated)
		# if state==24:
		# 	print(exponentiated)
		# 	exit()
		# print(prob)
	print(np.amax(q_array, 1).reshape((5,5)))
	exit()

	# Generate first trajectory of each
	trajectories_1 = list(generate_trajectories(1, gw, policy_1, initial, terminal))
	trajectories_2 = list(generate_trajectories(1, gw, policy_2, initial, terminal))

	solved = False

	obs = False

	while not solved:

		# Agent 2 observes agent 1s work
		# r = M.maxent_irl(gw, terminal, trajectories_1)

		# r = normalize(r)*5

		# print(r.reshape(5,5))

		# Compute the value function of this different reward
		v = value_iteration(gw.transition_prob, reward_1, 0.7)

		# Calculate Frechet Distance
		print(kendalltau(v, value_iteration(gw.transition_prob, reward_2, 0.7)))
		exit()
		plt.imshow(value_iteration(gw.transition_prob, reward_1, 0.7).reshape((5,5)))
		plt.show()
		plt.imshow(value_iteration(gw.transition_prob, reward_2, 0.7).reshape((5,5)))
		plt.show()
		print(f_d(value_iteration(gw.transition_prob, reward_1, 0.7), value_iteration(gw.transition_prob, reward_2, 0.7)))
		# print(signal.correlate(v, value_iteration(gw.transition_prob, reward_2, 0.9)))

		exit()
		if error <= er:
			solved = True
			print(f"Agent successful at demo {demo}")
			print(r.reshape((5,5)))
			break

		policy = find_policy(gw.transition_prob, (reward+(reward/2-r)), 0.9)

		# print(policy)
		# print(error)
		print("reward")
		print(r.reshape((5,5)))
		trajectories = trajectories + list(generate_trajectories(1, gw, policy, initial, terminal))

		# print(len(trajectories))
		demo+=1

def max_causal_ent_policy(transition_prob, reward, horizon, discount):
	nS = 25
	nA = 4
	V = np.zeros(nS)
	for i in range(horizon):
		Q = reward.reshape(nS, 1) + discount*(transition_prob*V).sum(2)
		V = sp_lse(Q, axis=1)
	print("Hello")
	print(V)
	return np.exp(Q-V.reshape(nS, 1))

if __name__ == "__main__":
	main()