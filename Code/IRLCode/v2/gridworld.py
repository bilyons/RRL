"""
Grid-world for Inverse Reinforcement Learning with Reflexive Component.

This pertains to the environment, containing all states and actions as well as
features of the environment for Max Ent IRL.
"""

import numpy as np
from itertools import product

class GridWorld:
	"""
	Deterministic Grid world is defined by the attribute size which
	defines both height and width.
	"""

	def __init__(self, size):

		self.size = size
		# 4 actions, down, up, right, left
		self.actions = [(1,0), (-1,0), (0, 1), (0, -1)]
		self.state_space = size**2
		self.action_space = len(self.actions)
		# Determine transition probabilities
		self.transition_prob = self._transition_prob_table()

	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.size, state // self.size

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.size + coord[0]

	def movement(self, coord, a):
		# Caps movement at world size
		new_y = coord[0]+self.actions[a][0]
		new_x = coord[1]+self.actions[a][1]
		new_y = max(0, min(self.size-1, new_y))
		new_x = max(0, min(self.size-1, new_x))
		return new_y, new_x

	def _transition_prob_table(self):
		"""
		Builds the internal probability transition table.
		Returns:
			The probability transition table of the form
				[state_from, state_to, action]
			containing all transition probabilities. The individual
			transition probabilities are defined by `self._transition_prob'.
		"""
		table = np.zeros(shape=(self.state_space, self.state_space, self.action_space))

		s1, s2, a = range(self.state_space), range(self.state_space), range(self.action_space)
		for s_from, s_to, a in product(s1, s2, a):
			table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

		return table

	def _transition_prob(self, s_from, s_to, a):
		"""
		Compute the transition probability for a single transition.
		Args:
			s_from: The state in which the transition originates.
			s_to: The target-state of the transition.
			a: The action via which the target state should be reached.
		Returns:
			The transition probability from `s_from` to `s_to` when taking
			action `a`.
		"""
		fx, fy = self.state_to_coordinate(s_from)
		tx, ty = self.state_to_coordinate(s_to)
		ax, ay = self.actions[a]

		# deterministic transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
				return 1.0

		# otherwise this transition is impossible
		return 0.0

	def __repr__(self):
		return "GridWorld(size={})".format(self.size)

def state_features(world):
	# Returns a feature matrix where each state is an individual feature
	# Identity matrix the size of the state space
	return np.identity(world.state_space)
