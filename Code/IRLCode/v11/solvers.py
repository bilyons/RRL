"""
Boltzmann action selection agent with value iteration as the solver
"""
import matplotlib 
matplotlib.use('Agg') 
import numpy as np
from tqdm import tqdm

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
		self.q_array = np.zeros((self.world.state_space, self.world.action_space))
		self.actions = np.arange(self.world.action_space)

	def action_selection(self, state):
		if np.random.rand()>self.epsilon:
			# print(self.q_array[state])
			prob = np.exp(np.divide(self.q_array[state], self.temp))
			# print(prob)
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			action = np.random.choice(self.actions, p=dist)
		else:
			action = np.random.randint(0,self.world.action_space)
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

	def train(self, episodes):
		start_epsilon_decay =1
		end_epsilon_decay = 0.1
		e_decay_value = (start_epsilon_decay - end_epsilon_decay)/episodes

		for eps in tqdm(range(episodes)):
			done = False
			t=0
			# Generates first state, not inclusive of end
			s = np.random.randint(0,(self.world.state_space-1))
			while s in self.world.terminals:
				s = np.random.randint(0,(self.world.state_space-1))
			while not done:
				action = self.action_selection(s)
				# convert new coord
				new_s = self.world.movement(s, action)
				# get reward 
				reward = self.world.rewards[new_s]
				# update values
				self.update_values(s, action, reward, new_s)
				if new_s in self.world.terminals:
					done = True
				s = new_s
			self.epsilon_decay(e_decay_value)

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
			discount = 0.80
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

	def train(self):
		start_epsilon_decay =1
		end_epsilon_decay = 0.1
		e_decay_value = (start_epsilon_decay - end_epsilon_decay)/episodes

		for eps in tqdm(range(episodes)):
			done = False
			t=0
			# Generates first state, not inclusive of end
			s = np.random.randint(0,(self.world.state_space-1))
			while s in terminals:
				s = np.random.randint(0,(self.world.state_space-1))
			while not done:
				action = self.action_selection(s)
				# convert new coord
				new_s = self.world.movement(s, action)
				# get reward 
				reward = self.world.rewards[new_s]
				# update values
				self.update_values(s, action, reward, new_s)
				if new_s in terminals:
					done = True
				s = new_s
			self.epsilon_decay(e_decay_value)
