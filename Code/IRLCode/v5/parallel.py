import numpy as np
import gridworld as W
import solvers as S
import plot as P
import optimizer as O
import trajectory as T
import maxent as M
import parallel as I
from copy import deepcopy

class Parallel(object):

	def __init__(self, world, rewards, terminals, expert_pol, max_length):
		self.world = world
		self.rewards = rewards
		self.terminals = terminals
		self.expert_pol = expert_pol
		self.max_length = max_length

	def RRL_run(self, solver, world, rewards, terminals, maxent):
		done = False
		trajectory = []
		s = np.random.randint(0, world.state_space)
		# print(maxent)
		while s in terminals:
			s = np.random.randint(0, world.state_space)
		while not done:
			action = solver.action_selection(s)
			# convert to coord
			new_s = world.movement(s, action)
			# get reward
			reward = rewards[new_s]
			# compute effective action
			eff_a = world.effective_action(action, s, new_s)
			action = eff_a
			if new_s in terminals:
				done = True
				reward += world.reflexive_reward(new_s, maxent, rewards)
			# print(reflexive[1], action, reward)

			# update values
			solver.update_values(s, action, reward, new_s)
			trajectory += [(s, action, new_s)]

			s = new_s
		return T.Trajectory(trajectory)

	def maxent(self,world, terminals, trajectories, feat_count, init_count, ep_number):
		"""
		Max ent irl
		"""
		# Get features from world
		features = W.state_features(self.world)

		# Initialization parameters
		init = O.Constant(1.0)

		# Optimization strategy
		optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

		# IRL
		reward, feat_count, init_count = M.irl(world, features, terminals, trajectories, 
			optim, init, feat_count, init_count, ep_number)

		return reward, feat_count, init_count

	def paralleled(self, n_run):
		# np.random.seed()
		# print("hi")
		world = deepcopy(self.world)
		rewards = deepcopy(self.rewards)
		terminals = deepcopy(self.terminals)
		expert_pol = deepcopy(self.expert_pol)
		episodes = deepcopy(self.max_length)
		delta = 0.05

		solver = S.BoltzmannAgent(world, 0.1, 0.90, 0.1)
		solver.q_array = np.copy(expert_pol)

		p_solver = deepcopy(solver)

		feat_count = np.zeros(world.state_space)
		init_count = np.zeros(world.state_space)
		reward_maxent = rewards

		for ep in range(episodes):
			new_trajectory = (self.RRL_run(p_solver, world, rewards, terminals, reward_maxent))
			
			reward_maxent, feat_count, init_count = self.maxent(world, terminals, new_trajectory, 
				feat_count, init_count, ep)
			# print(n_run, ep)
			# if n_run == 0:
				# print(id(world))
			if ep==100:
				print("Run: ", n_run, " has reached convergence threshold")
			if ep>100 and reward_maxent[world.size-1]*0.65 - delta < reward_maxent[0] and reward_maxent[0] < reward_maxent[world.size-1]*0.65 + delta:
				# if run%2 == 0:
				print("Success")
				return [n_run, 1, ep, reward_maxent]
		print(n_run, ": completed")	
		return [n_run, 0, ep, reward_maxent]