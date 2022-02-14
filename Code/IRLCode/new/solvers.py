"""
Implements MDP solvers

Billy Lyons, 2021
billy.lyons@ed.ac.uk

As it stands, this is just a Boltzmann agent, in future add other methods
"""

import numpy as np
from tqdm import tqdm
import trajectory as T

class BoltzmannAgent:
	def __init__(self, world, t_length, temp=0.1, gamma=0.9, lr=0.1, epsilon=1):
		self.world = world
		self.t_length = t_length
		self.temp = temp
		self.gamma = gamma
		self.lr = lr
		self.epsilon = epsilon

		# Generate Q Table and Actions
		self.q_array = np.zeros((world.n_states, world.n_actions))
		self.actions = np.arange(world.n_actions)

	def action_selection(self, state):
		if np.random.rand()>self.epsilon:
			# print(self.q_array[state])
			prob = np.exp(np.divide(self.q_array[state], self.temp))
			# print(self.q_array[state], self.temp)
			# print(prob)
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			action = np.random.choice(self.actions, p=dist)
		else:
			action = np.random.randint(0,self.world.n_actions)
		return action

	def return_policy(self):
		policy = np.zeros((self.world.n_states, self.world.n_actions))
		for s in range(self.world.n_states):
			prob = np.exp(np.divide(self.q_array[s], self.temp))
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			policy[s] = dist
		return policy

	def return_value(self):
		value = np.zeros(self.world.n_states)
		for s in range(self.world.n_states):
			value[s] = np.max(self.q_array[s,:])
		return value

	def update_values(self, old_state, action, reward, new_state):
		cur_q = self.q_array[old_state][action]
		max_future_q = np.max(self.q_array[new_state])
		new_q = cur_q + self.lr*(reward + self.gamma*max_future_q - cur_q)
		self.q_array[old_state][action] = new_q

	def epsilon_decay(self, decay_value):
		self.epsilon -= decay_value

	def train(self, episodes):
		start_epsilon_decay = 1
		end_epsilon_decay = 0.1
		e_decay_value = (start_epsilon_decay - end_epsilon_decay)/episodes

		for eps in tqdm(range(episodes)):
			t=0
			done = False
			# Generates first state, not inclusive of end
			s = np.random.randint(0,(self.world.n_states-1))
			while self.world.is_goal(s):
				s = np.random.randint(0,(self.world.n_states-1))
			while t < self.t_length:
			# while not done:
				action = self.action_selection(s)
				# convert new coord
				new_s = self.world.movement(s, action)
				# get reward 
				reward = self.world.rewards[new_s]
				# update values
				self.update_values(s, action, reward, new_s)
				s = new_s
				if self.world.is_goal(s):
					done = True
				t+=1
			self.epsilon_decay(e_decay_value)

	def run(self, rr=None, start_state=None):
		t=0
		done = False
		trajectory = []
		# Generates first state, not inclusive of end
		if start_state == None:
			s = np.random.randint(0,(self.world.n_states-1))
			while self.world.is_goal(s):
				s = np.random.randint(0,(self.world.n_states-1))
		else:
			s = start_state
		while t < self.t_length:
		# while not done:
			# print(s)
			action = self.action_selection(s)
			# convert new coord
			new_s = self.world.movement(s, action)
			# get reward 
			reward = self.world.rewards[new_s]
			if rr is not None:
				# print(rr)
				# exit()
				reward += self.world.reflexive_reward(new_s, rr)
				# print(new_s, reward)
			# update values
			self.update_values(s, action, reward, new_s)
			# Add transition
			trajectory += [(s, action, new_s)]
			s = new_s
			if self.world.is_goal(s):
					done = True
			t+=1
		return T.Trajectory(trajectory)