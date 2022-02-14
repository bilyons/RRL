"""
Grid-world for Inverse Reinforcement Learning with Reflexive Component.

This pertains to the environment, containing all states and actions as well as
features of the environment for Max Ent IRL.
"""

import numpy as np
from itertools import product

class LineWorld:

	def __init__(self, size, p_slip):
		self.size = size

		# 2 actions, left and right
		self.actions = [-1, 1]
		self.state_space = size
		self.action_space = len(self.actions)

		self.p_slip = p_slip
		# Determine trasition probabilities
		self.transition_prob = self._transition_prob_table()

	def movement(self, state, a):
		# Move left and right
		return np.random.choice(self.state_space, p=self.transition_prob[state,:, a])

	def effective_action(self, action, old_state, new_state):
		if new_state-old_state > 0:
			# Moved right
			eff_a = 1
		else:
			# Moved left
			eff_a = 0
		return eff_a

	def reflexive_reward(self, state, maxent_array, rewards):
		des_diff = rewards[0]
		diff = maxent_array[0]/maxent_array[self.state_space-1]
		ref = des_diff - diff

		if ref > 0 and state == 0:
			return 1.0+ref
		elif ref < 0 and state > 0:
			return 1.0+np.abs(ref)
		else:
			return 0.0

	# This shouldn't be here but I need it to not have to mess around with other code
	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state 

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord

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

		# print(table)

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

		a = self.actions[a]
		# deterministic transition defined by action
		if s_from + a == s_to:
			return 1.0 - self.p_slip + self.p_slip/ self.action_space

		if abs(s_from - s_to) == 1:
			return self.p_slip/self.action_space

		# Below doesn't matter as edges are terminal but for completeness
		if s_from == s_to:
			# Intended move over edge
			if not 0 <= s_from + a < self.size:

				return 1.0 - self.p_slip + self.p_slip/self.action_space

			# Unintentional
			if not 0 < s_from < self.size-1:
				return self.p_slip/self.action_space

		# otherwise this transition is impossible
		return 0.0

	def __repr__(self):
		return "LineWorld(size={})".format(self.size)

class GridWorld:
	"""
	Deterministic Grid world is defined by the attribute size which
	defines both height and width.
	"""

	def __init__(self, size, p_slip):

		self.size = size
		# 4 actions, down, up, right, left
		self.actions = [(1,0), (-1,0), (0, 1), (0, -1)]
		self.state_space = size**2
		self.action_space = len(self.actions)

		self.p_slip = p_slip
		# Determine transition probabilities
		self.transition_prob = self._transition_prob_table()

	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.size, state // self.size

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.size + coord[0]

	def movement(self, state, a):
		# Caps movement at world size
		return np.random.choice(self.state_space, p=self.transition_prob[state,:, a])

	def effective_action(self, action, old_state, new_state):
		if new_state-old_state == 0:
			eff_a = action
		elif new_state-old_state == self.size:
			# Moved down
			eff_a = 0
		elif new_state-old_state == -self.size:
			# Moved up
			eff_a = 1
		elif new_state-old_state == 1:
			# Moved right
			eff_a = 2
		elif new_state-old_state == -1:
			# Moved left
			eff_a = 3
		else:
			print("Fuck up")
			print(action, old_state, new_state)
			exit()
		return eff_a

	def reflexive_reward(self, state, maxent_array, rewards):
		des_diff = rewards[(self.state_space-self.size)]
		diff = maxent_array[(self.state_space-self.size)]/maxent_array[self.state_space-1]
		ref = des_diff - diff
		if ref > 0 and state == (self.state_space-self.size):
			return 1.0+ref
		elif ref < 0 and state > (self.state_space-self.size):
			return 1.0+np.abs(ref)
		else:
			return 0.0

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

		# print(table)

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
		# intended transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0 - self.p_slip + self.p_slip / self.action_space

		# we can slip to all neighboring states
		if abs(fx - tx) + abs(fy - ty) == 1:
			return self.p_slip / self.action_space

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			# intended move over an edge
			if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
				# double slip chance at corners
				if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
					return 1.0 - self.p_slip + 2.0 * self.p_slip / self.action_space

				# regular probability at normal edges
				return 1.0 - self.p_slip + self.p_slip / self.action_space

			# double slip chance at corners
			if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
				return 2.0 * self.p_slip / self.action_space

			# single slip chance at edge
			if not 0 < fx < self.size - 1 or not 0 < fy < self.size - 1:
				return self.p_slip / self.action_space

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def __repr__(self):
		return "GridWorld(size={})".format(self.size)

def state_features(world):
	# Returns a feature matrix where each state is an individual feature
	# Identity matrix the size of the state space
	return np.identity(world.state_space)
