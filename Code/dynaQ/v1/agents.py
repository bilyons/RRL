"""
Implements MDP solvers

Billy Lyons, 2021
billy.lyons@ed.ac.uk

As it stands, this is just a Boltzmann agent, in future add other methods
"""

import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import trajectory as T

class BoltzmannAgent:
	def __init__(self, world, temp, gamma, lr, planning_steps):
		self.world = world
		self.temp = temp
		self.gamma = gamma
		self.lr = lr
		self.N = planning_steps
		self.epsilon = 1

		# Generate Q table and actions
		self.q_array = np.zeros((world.n_states, world.n_actions))
		self.actions = np.arange(world.n_actions)

		# Model dictionary 
		self.t_model = np.zeros((world.n_states, world.n_states, world.n_actions))
		self.r_model = np.zeros((world.n_states, world.n_actions))
		self.tau = np.zeros((world.n_states, world.n_actions))
		self.count = np.zeros((world.n_states, world.n_actions))

	def act(self, state):
		if np.random.rand() > self.epsilon:
			prob = np.exp(np.divide(self.q_array[state], self.temp))
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			action = np.random.choice(self.actions, p=dist)
		else:
			action = np.random.randint(0, self.world.n_actions)
		return action

	def update_model(self, s, a, s_new, r):
		self.r_model[s,a] += r # adds all rewards from this action to the transition
		self.t_model[s,s_new,a] += 1 # counts where you end up after
		self.tau += 1
		self.tau[s][a] = 0
		self.count[s][a] += 1

	def plan(self, irl_output):
		for n in range(self.N):
			p = random.choice(np.argwhere(np.array(self.count)>0))
			s, act = p[0], p[1]
			# Perceived new state
			new_s = np.random.choice(self.world.n_states, p = self.world.transition_prob[s,:,act])
			# Perceived reward
			if irl_output is not None:
				r =self.world.rewards[new_s]+irl_output[new_s]
				if new_s == s:
					r-=1
			else:
				r = (self.r_model[s][act]/self.count[s][act])
			self.update_values(s, act, r, new_s)

	def update_values(self, old_state, action, reward, new_state):
		cur_q = self.q_array[old_state][action]
		max_future_q = np.max(self.q_array[new_state])
		new_q = cur_q + self.lr*(reward + self.gamma*max_future_q - cur_q)
		self.q_array[old_state][action] = new_q

	def epsilon_decay(self, decay_value):
		self.epsilon -= decay_value

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
		return value.reshape((self.world.full_size, self.world.full_size))

	def epsilon_decay(self, decay_value):
		self.epsilon -= decay_value

	def train(self, episodes, irl_output):
		start_epsilon_decay =1
		end_epsilon_decay = 0.1
		e_decay_value = (start_epsilon_decay - end_epsilon_decay)/episodes

		for eps in tqdm(range(episodes)):
			done = False
			# Generates first state, not inclusive of end
			# s = np.random.randint(0,(self.world.n_states-1))
			# while self.world.is_goal(s):
			# 	s = np.random.randint(0,(self.world.n_states-1))
			s=2
			while not done:
				action = self.act(s)
				# convert new coord
				new_s = self.world.movement(s, action)
				# get reward 
				reward = self.world.rewards[new_s]
				if new_s == s:
					reward-=1
				# update values
				self.update_values(s, action, reward, new_s)
				self.update_model(s, action, new_s, reward)
				self.plan(irl_output)
				s = new_s
				if self.world.is_goal(s):
					done = True
			self.epsilon_decay(e_decay_value)

	def run(self,irl_output):
		t=0
		done = False
		trajectory = []
		# Generates first state, not inclusive of end
		t=0
		done = False
		# Generates first state, not inclusive of end
		# s = np.random.randint(0,(self.world.n_states-1))
		# while self.world.is_goal(s):
		# 	s = np.random.randint(0,(self.world.n_states-1))
		s=2
		while not done:
			action = self.act(s)
			# convert new coord
			new_s = self.world.movement(s, action)
			# get reward 
			reward = self.world.rewards[new_s]
			if new_s == s:
				reward-=1
			# update values
			self.update_values(s, action, reward, new_s)
			self.update_model(s, action, new_s, reward)
			self.plan(irl_output)
			trajectory += [(s, action, new_s)]
			s = new_s
			if self.world.is_goal(s):
				done = True
			t+=1
		return T.Trajectory(trajectory)

	def cum_reward(self, ep, t, reward):
		if ep == 0:
			self.t_steps = []
			self.rewards = []
		self.t_steps.append(t)
		self.rewards.append(reward)
