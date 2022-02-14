import numpy as np
import random
from itertools import chain
"""
This file takes a policy in the form of a QArray, and generates a list of trajectories
for use in max ent IRL.
"""

class Trajectory:

	"""
	A trajectory consists of:
	States, Actions and Outcomes in the form of a tuple (state_origin, action, state_end)
	state_end should match state_origin of the next segment of the trajectory
	"""

	def __init__(self, transitions):
		self._t = transitions

	def transitions(self):
		"""
		Returns all transitions as an array of tuples
		(state_origin, action, state_end)
		"""
		return self._t

	def __repr__(self):
		return "Trajectory({})".format(repr(self._t))

	def __str__(self):
		return "{}".format(self._t)

	def states(self):
		"""
		The states visited in this trajectory.
		Returns:
			All states visited in this trajectory as iterator in the order
			they are visited. If a state is being visited multiple times,
			the iterator will return the state multiple times according to
			when it is visited.
		"""
		return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

def generate_trajectory(world, policy, stochastic, terminals):
	"""
	Generate a single trajectory
	"""

	state = random.choice([x for x in range(world.state_space) if x not in terminals])
	state_coord = world.state_to_coordinate(state)
	trajectory = []
	actions = [0,1,2,3]
	
	while state not in terminals:
		if stochastic == 0:
			action = np.argmax(policy[state])
		else:
			action = np.random.choice(actions, p=policy[state])

		new_coord = world.movement(state_coord, action)

		new_state = world.coordinate_to_state(new_coord)

		trajectory += [(state, action, new_state)]

		state = new_state
		state_coord = new_coord
	return Trajectory(trajectory)

def generate_trajectories(n, world, policy, stochastic, terminals):
	
	return (generate_trajectory(world, policy, stochastic, terminals) for _ in range(n))

def add_trajectories(trajectories_list, trajectory):
	trajecories_list = trajectories_list + trajectory
	return trajectories_list + trajectory


