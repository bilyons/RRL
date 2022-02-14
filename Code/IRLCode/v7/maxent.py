from itertools import product

import numpy as np

def maxent_irl(world, solver, trajectory, init_count, feat_count, epochs, ep_number):

	"""
	Find reward function for given trajectoies

	Args:
		world: take in the world for the transition matrix, terminals, features etc.
		solver: whatever is doing the solving
		trajectory: current trajectory
		init + feat counts: counts at last time step as these are just integers
		epochs: number of steps to do gradient descent

	Output:
		reward function
	"""

	n_states = world.state_space
	n_actions = world.action_space
	features = world.features
	_, n_features = features.shape

	p_transition = world.p_transition

	# Initialise weights
	alpha = np.random.uniform(size=(n_states,))

	# Calculate feature expectations
	feature_expectations = feature_expectation_updatable(features, feat_count, init_count, ep_number)