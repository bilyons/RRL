"""
Implements gridworld MDP

Billy Lyons, 2021
billy.lyons@ed.ac.uk

Adapted from Matthew Alger: https://github.com/MatthewJA/Inverse-Reinforcement-Learning
"""

import numpy as np
from itertools import product

class GridWorld(object):
	"""
	Gridworld environment
	"""

	def __init__(self, size, rewards, p_slip, discount):
		"""
		input:
			size: grid size of a side, envs are square, resulting NxN
			terminals: list of terminating states
			rewards: array of rewards in the state space
			p_slip: traditionally "wind", change of slipping during transition
		"""

		self.actions = [(1,0), (-1,0), (0, 1), (0, -1)]
		self.n_actions = len(self.actions)
		self.n_states = size**2
		self.size = size
		self.p_slip = p_slip
		self.discount = discount
		self.rewards = rewards
		self.features = state_features(self)

		# Construct probability array
		self.transition_prob = self._transition_prob_table()

	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.size, state // self.size

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.size + coord[0]

	def _transition_prob_table(self):
		"""
		Builds the internal probability transition table.
		Returns:
			The probability transition table of the form
				[state_from, state_to, action]
			containing all transition probabilities. The individual
			transition probabilities are defined by `self._transition_prob'.
		"""
		table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

		s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
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
		
		Note: over the weekend, do away with state to and state from
		Consider modular addition and you should be able to extend to 8
		actions with minimal effort

		"""
		fx, fy = self.state_to_coordinate(s_from)
		tx, ty = self.state_to_coordinate(s_to)
		ax, ay = self.actions[a]

		# deterministic transition defined by action
		# intended transition defined by action
		if fx + ax == tx and fy + ay == ty:
			return 1.0 - self.p_slip + self.p_slip / self.n_actions

		# we can slip to all neighboring states
		if abs(fx - tx) + abs(fy - ty) == 1:
			return self.p_slip / self.n_actions

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			# intended move over an edge
			if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
				# double slip chance at corners
				if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
					return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions

				# regular probability at normal edges
				return 1.0 - self.p_slip + self.p_slip / self.n_actions

			# double slip chance at corners
			if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
				return 2.0 * self.p_slip / self.n_actions

			# single slip chance at edge
			if not 0 < fx < self.size - 1 or not 0 < fy < self.size - 1:
				return self.p_slip / self.n_actions

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def state_features(self):
		# Returns a feature matrix where each state is an individual feature
		# Identity matrix the size of the state space
		return np.identity(self.n_states)

	def reward(self, state):
		return self.rewards[state]

	def reflexive_reward(self, state, maxent_array):
		# print(state)
		# print(maxent_array)
		diff = np.abs(np.abs(maxent_array[self.n_states-self.size] - maxent_array[self.n_states-1]) - 0.5)
		if state == self.n_states-1:
			proportion = self.rewards[state]/(self.rewards[state] - 0.5)
			# If small reward bigger and at small, no reflexive
			if maxent_array[state] < proportion*maxent_array[self.n_states-self.size]:
				# big reward too small
				return 2*diff
			else:
				# Big reward correct or larger
				return -2*diff
		elif state == self.n_states-self.size:
			# print("I am in the right place")
			proportion = self.rewards[state]/(self.rewards[self.n_states -1])
			# If small reward bigger and at small, no reflexive
			if maxent_array[state] < proportion*maxent_array[self.n_states-1]:
				# big reward too small
				return 2*diff
			else:
				# Big reward correct or larger
				return -2*diff
		else:
			return 0.0

	def is_goal(self, state):
		if self.rewards[state] > 0:
			return True
		else:
			return False

	def movement(self, state, action):
		return np.random.choice(self.n_states, p=self.transition_prob[state,:,action])

def state_features(world):
	# Returns a feature matrix where each state is an individual feature
	# Identity matrix the size of the state space
	return np.identity(world.n_states)