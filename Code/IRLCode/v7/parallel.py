import matplotlib 
matplotlib.use('Agg') 
import numpy as np
import trajectory as T
# import optimizer as O
import maxent as M
import gridworld as W
from tqdm import tqdm
from copy import deepcopy

class Parallel:

	def __init__(self, world, solver, episodes):
		self.world = world
		self.solver = solver
		self.episodes = episodes

	def RRL_run(self, solver, world, maxent):
		done = False
		trajectory = []
		s = np.random.randint(0, world.state_space)
		# print(maxent)
		solver.epsilon = 0
		while s in world.terminals:
			s = np.random.randint(0, world.state_space)
		while not done:
			action = solver.action_selection(s)
			# convert to coord
			new_s = world.movement(s, action)
			# get reward
			reward = world.rewards[new_s]
			# compute effective action
			# eff_a = world.effective_action(action, s, new_s)
			# action = eff_a

			if new_s in world.terminals:
				done = True
				reward += world.reflexive_reward(new_s, maxent, world.rewards)
				# print(reward)
			# print(reflexive[1], action, reward)

			# update values
			solver.update_values(s, action, reward, new_s)
			trajectory += [(s, action, new_s)]
			
				#print(solver.epsilon, eps)
			s = new_s
		return T.Trajectory(trajectory)

	def maxent(self, world, trajectories, feat_count, init_count, ep_number):
		"""
		Max ent irl
		"""
		# Get features from world
		features = W.state_features(world)

		# Initialization parameters
		init = O.Constant(1.0)

		# Optimization strategy
		optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

		# IRL
		reward, feat_count, init_count = M.maxent_irl(world, features, world.terminals, trajectories, 
			optim, init, feat_count, init_count, ep_number)

		return reward, feat_count, init_count

	def parallel(self, n_run):
		delta = 0.05
		solver = deepcopy(self.solver)
		world = deepcopy(self.world)

		feat_count = np.zeros(world.state_space)
		init_count = np.zeros(world.state_space)
		reward_maxent = world.rewards

		for ep in range(self.episodes):
			new_trajectory = (self.RRL_run(solver, world, reward_maxent))
			# print(solver.q_array)
			reward_maxent, feat_count, init_count = self.maxent(world, new_trajectory, 
				feat_count, init_count, ep)
			# reward_causal= maxent_causal(world, terminals, new_trajectory, 
			# 	feat_count, init_count, ep,discount=0.8)

			# if run%2 == 0:
			# print(world.dif)
			if ep>50 and (reward_maxent[world.state_space-1]*(1-world.dif) - delta) < reward_maxent[world.state_space-world.size] and (reward_maxent[world.state_space-world.size] < reward_maxent[world.state_space-1]*(1-world.dif) + delta):
				# if run%2 == 0:
				# rrl.append(1)
				reward_maxent = reward_maxent/reward_maxent[world.size-1]
				# else:
				# 	non_rrl.append(1)
				return [n_run, 1, ep, reward_maxent]

			elif ep==499:
				reward_maxent = reward_maxent/reward_maxent[world.size-1]
				return [n_run, 0, ep, reward_maxent]
		