"""
Boltzmann action selection agent with value iteration as the solver
"""
import numpy as np

class BoltzmannAgent:
	def __init__(self, world, temp, discount=None, lr=None, epsilon=None):
		self.world = world
		self.temp = temp

		if epsilon is None:
			epsilon = 1
		self.epsilon = epsilon
		if lr is None:
			lr = 0.1
		self.lr = lr
		if discount is None:
			discount = 0.95
		self.discount = discount

		# Generate Q Table and Actions
		self.q_array = np.ones((self.world.size**2, self.world.action_space))
		self.actions = np.arange(self.world.action_space)

	def action_selection(self, state):
		if np.random.rand()>self.epsilon:
			prob = np.exp(np.divide(self.q_array[state], self.temp))
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			action = np.random.choice(self.actions, p=dist)
		else:
			action = np.random.randint(0,4)
		return action

	def return_policy(self):
		policy = np.zeros((self.world.state_space, self.world.action_space))
		for s in range(self.world.state_space):
			prob = np.exp(np.divide(self.q_array[s], self.temp))
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			policy[s] = dist
		return policy

	def update_values(self, old_state, action, reward, new_state):
		cur_q = self.q_array[old_state][action]
		max_future_q = np.max(self.q_array[new_state])
		new_q = cur_q + self.lr*(reward + self.discount*max_future_q - cur_q)
		self.q_array[old_state][action] = new_q

	def epsilon_decay(self, decay_value):
		self.epsilon -= decay_value

class GreedyAgent:
	def __init__(self, world, temp, discount=None, lr=None, epsilon=None):
		self.world = world
		self.temp = temp

		if epsilon is None:
			epsilon = 1
		self.epsilon = epsilon
		if lr is None:
			lr = 0.1
		self.lr = lr
		if discount is None:
			discount = 0.95
		self.discount = discount

		# Generate Q Table and Actions
		self.q_array = np.zeros((self.world.size**2, self.world.action_space))
		self.actions = np.arange(self.world.action_space)

	def action_selection(self, state):
		if np.random.rand()>self.epsilon:
			action = np.argmax(self.q_array[state])
		else:
			action = np.random.randint(0,4)
		return action

	def update_values(self, old_state, action, reward, new_state):
		cur_q = self.q_array[old_state][action]
		max_future_q = np.max(self.q_array[new_state])
		new_q = cur_q + self.lr*(reward + self.discount*max_future_q - cur_q)
		self.q_array[old_state][action] = new_q

	def epsilon_decay(self, decay_value):
		self.epsilon -= decay_value