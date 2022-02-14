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

	def __init__(self, full_size, p_slip, r_dif):#, spawn_size):
		"""
		input:
			size: grid size of a side, envs are square, resulting NxN
			terminals: list of terminating states
			rewards: array of rewards in the state space
			p_slip: traditionally "wind", change of slipping during transition
		"""

		self.actions = [(1,0), (-1,0), (0, 1), (0, -1)]#, (0, 0)]
		self.n_actions = len(self.actions)
		self.n_states = full_size**2
		self.full_size = full_size
		self.p_slip = p_slip
		self.r_dif = r_dif
		self.terminals = [20, 24]
		# self.spawn_size = spawn_size
		# self.offset = int((self.full_size-self.spawn_size)/2)

		self.min_ = self.n_states - self.full_size #int(self.offset)
		self.max_ = self.n_states - 1 #int((self.offset+self.spawn_size))

		self.features = state_features(self)

		# Construct probability array
		self.transition_prob = self._transition_prob_table()

		self.rewards = self.create_rewards()

	def state_to_coordinate(self, state):
		# Converts a state from s in size**2 to (y,x) coordinate
		return state % self.full_size, state // self.full_size

	def coordinate_to_state(self, coord):
		# Converts a coordinate represented state to full index
		return coord[1]*self.full_size + coord[0]

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

	# def _transition_prob(self, s_from, s_to, a):
	# 	"""
	# 	Compute the transition probability for a single transition.
	# 	Args:
	# 		s_from: The state in which the transition originates.
	# 		s_to: The target-state of the transition.
	# 		a: The action via which the target state should be reached.
	# 	Returns:
	# 		The transition probability from `s_from` to `s_to` when taking
	# 		action `a`.
		
	# 	Note: over the weekend, do away with state to and state from
	# 	Consider modular addition and you should be able to extend to 8
	# 	actions with minimal effort

	# 	Adding in stationary action. Comment

	# 	"""
	# 	fx, fy = self.state_to_coordinate(s_from)
	# 	tx, ty = self.state_to_coordinate(s_to)
	# 	ax, ay = self.actions[a]

	# 	# deterministic transition defined by action
	# 	# intended transition defined by action 

	# 	# If choose to stay, remains fx+ax = fx = tx, fy+ay = fy = ty
	# 	# will work for stationary
	# 	if fx + ax == tx and fy + ay == ty:
	# 		# If we are choosing to remain still and at an edge, increased slip chance
	# 		if ax == 0 and ay== 0:
	# 			if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
	# 				# We are in a corner choosing to remain. 3 slips keep me here, 2 kick me out
	# 				return 1.0 - self.p_slip + 3.0 * self.p_slip / self.n_actions
	# 			if not 0 < fx < self.full_size - 1 or not 0 < fy < self.full_size - 1:
	# 				# If I am at an edge choosing to remain, 2 slips keep me here, 3 kick me out
	# 				return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions
	# 			else:
	# 				return 1.0 - self.p_slip + self.p_slip / self.n_actions			
	# 		else:
	# 			return 1.0 - self.p_slip + self.p_slip / self.n_actions

	# 	# we can slip to all neighboring cardinal states
	# 	# Need to be able to slip to self
	# 	if abs(fx - tx) + abs(fy - ty) == 1:
	# 		return self.p_slip / self.n_actions

	# 	# we can stay at the same state if we would move over an edge
	# 	if fx == tx and fy == ty:

	# 		# intended move over an edge
	# 		if not 0 <= fx + ax < self.full_size or not 0 <= fy + ay < self.full_size:
	# 			# triple slip chance at corners
	# 			if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
	# 				return 1.0 - self.p_slip + 3.0 * self.p_slip / self.n_actions

	# 			# regular probability at normal edges
	# 			return 1.0 - self.p_slip + 2.0*self.p_slip / self.n_actions

	# 		# triple slip chance at corners - into and remain
	# 		if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
	# 			return 3.0 * self.p_slip / self.n_actions

	# 		# double slip at edge - into and remain
	# 		if not 0 < fx < self.full_size - 1 or not 0 < fy < self.full_size - 1:
	# 			return 2.0*self.p_slip / self.n_actions

	# 		# otherwise we can remain in the same state
	# 		return self.p_slip / self.n_actions

	# 	# otherwise this transition is impossible
	# 	return 0.0
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
			return 1.0 - self.p_slip + self.p_slip / self.n_actions

		# we can slip to all neighboring states
		if abs(fx - tx) + abs(fy - ty) == 1:
			return self.p_slip / self.n_actions

		# we can stay at the same state if we would move over an edge
		if fx == tx and fy == ty:
			# intended move over an edge
			if not 0 <= fx + ax < self.full_size or not 0 <= fy + ay < self.full_size:
				# double slip chance at corners
				if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
					return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions

				# regular probability at normal edges
				return 1.0 - self.p_slip + self.p_slip / self.n_actions

			# double slip chance at corners
			if not 0 < fx < self.full_size - 1 and not 0 < fy < self.full_size - 1:
				return 2.0 * self.p_slip / self.n_actions

			# single slip chance at edge
			if not 0 < fx < self.full_size - 1 or not 0 < fy < self.full_size - 1:
				return self.p_slip / self.n_actions

			# otherwise we cannot stay at the same state
			return 0.0

		# otherwise this transition is impossible
		return 0.0

	def state_features(self):
		# Returns a feature matrix where each state is an individual feature
		# Identity matrix the size of the state space
		return np.identity(self.n_states)

	def create_rewards(self):
		# y = int(np.floor(self.spawn_size/2))
		# rewards = np.zeros(self.full_size**2)
		# self.large_r = self.coordinate_to_state((self.offset,self.offset+y))
		# rewards[self.large_r] = 1
		# self.small_r = self.coordinate_to_state((self.offset + self.spawn_size-1,self.offset+y))
		# rewards[self.small_r] = 1.0-self.r_dif

		rewards = np.zeros((self.n_states))
		rewards[self.min_] = 1.0 - self.r_dif
		rewards[self.max_] = 1.0
		return rewards

	def reward(self, state):
		return self.rewards[state]

	def reflexive_reward(self, state, maxent_array):
		# print(state)
		# print(maxent_array)
		diff = np.abs(np.abs(maxent_array[self.n_states-self.full_size] - maxent_array[self.n_states-1]) - self.r_dif)
		if state == self.large_r:
			proportion = self.rewards[state]/(self.rewards[state] - self.r_dif)
			# If small reward bigger and at small, no reflexive
			if maxent_array[state] < proportion*maxent_array[self.n_states-self.full_size]:
				# big reward too small
				return diff
			else:
				# Big reward correct or larger
				return diff
		elif state == self.small_r:
			# print("I am in the right place")
			proportion = self.rewards[state]/(self.rewards[self.n_states -1])
			# If small reward bigger and at small, no reflexive
			if maxent_array[state] < proportion*maxent_array[self.n_states-1]:
				# big reward too small
				return diff
			else:
				# Big reward correct or larger
				return -diff
		else:
			return 0.0

	def is_goal(self, state):
		if self.rewards[state] > 0:
			return True
		else:
			return False

	# def spawn(self):
	# 	# Function for spawning agent in a valid state
	# 	x = np.random.randint(self.min_, self.max_)
	# 	y = np.random.randint(self.min_, self.max_)

	# 	# print(x,y)
	# 	s = self.coordinate_to_state((x,y))
	# 	# print(s)
	# 	while self.is_goal(s):
	# 		x = np.random.randint(self.min_, self.max_)
	# 		y = np.random.randint(self.min_, self.max_)
	# 		s = self.coordinate_to_state((x,y))
	# 	return s

	def movement(self, state, action):
		# print(state, action)
		return np.random.choice(self.n_states, p=self.transition_prob[state,:,action])

def state_features(world):
	# Returns a feature matrix where each state is an individual feature
	# Identity matrix the size of the state space
	return np.identity(world.n_states)