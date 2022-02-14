import sys, os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

import gridworld
import value_iteration
import trajectory as T
import irl as I

def main(grid_size, discount, learning_rate, wind, r_dif):
	"""
	"""

	# Create world
	gw = gridworld.Gridworld(grid_size, wind, discount, r_dif)

	# Get ground truth rewards
	ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

	# Generate policy
	policy = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
											ground_r, gw.discount, stochastic=True)

	policy_exec = T.stochastic_policy_adapter(policy)
	# Generate trajectories
	initial = np.zeros(gw.n_states)
	initial[2] = 1.0

	terminals = [20,24]
	# Set prelim to 5
	trajectories = list(T.generate_trajectories(5, gw, policy_exec, initial, terminals))

	# New update
	solved = False

	# Start count
	demo = 5
	while not solved:

		# Reward
		rewards = I.m_irl(gw, trajectories, terminals)

		# Compute Pearson's Rank
		error = stats.pearsonr(ground_r, rewards)

		if error[0] >= 0.95:
			solved = True
			print(f"Successful at demo {demo}")
			break

		# Learn New Policy
		policy = value_iteration.find_policy(gw.n_states, gw.n_actions, gw.transition_probability,
												(ground_r-rewards), gw.discount, stochastic=True)

		policy_exec = T.stochastic_policy_adapter(policy)

		trajectories = trajectories + list(T.generate_trajectories(1, gw, policy_exec, initial, terminals))

		demo += 1


if __name__ == '__main__':
	main(5, 0.9, 0.01, 0.0, 0.5)
