import matplotlib 
matplotlib.use('Agg') 
import numpy as np
import trajectory as T
import irl as I
import gridworld as W
from tqdm import tqdm
from copy import deepcopy
from time import sleep
class Parallel:

	def __init__(self, world, solver, episodes):
		self.world = world
		self.solver = solver
		self.episodes = episodes

	def RRL(self, solver, world, num_trajectories, r_bool, irl_array):
		# Takes in the solver, world, and a desired number of trajectories to add in the batch
		# Outputs the new trajectories
		trajectories = []
		# For number of trajectories to add
		for t in range(num_trajectories):
			# Initialise empty trajectory
			trajectory = []
			# Set done to false for solve
			done = False
			# Initialise
			s = np.random.randint(0, world.state_space)
			while s in world.terminals: # Catches initialising in terminal. Not allowed
				s = np.random.randint(0, world.state_space) 
			# Episode
			while not done:
				# Draw action
				action = solver.action_selection(s)
				# Get new state
				new_s = world.movement(s, action)

				# Get reward
				reward = world.rewards[new_s]

				if new_s in world.terminals:
					done = True
					if r_bool:
						reward += world.reflexive_reward(new_s, irl_array, world.rewards)
						# print("new_s", new_s)
						# print("reward", reward)
					# print(reward)
				solver.update_values(s, action, reward, new_s)
				trajectory += [(s, action, new_s)]

				s = new_s

			trajectory = T.Trajectory(trajectory)

			trajectories.append(trajectory)

		return trajectories

	def add_trajectories(self, solver, old_trajs, world, new_number, feat_count, init_count, r_bool, irl_array):
		new_trajs = self.RRL(solver, world, new_number, r_bool, irl_array)
		combined_trajs = old_trajs + new_trajs
		feat_count = I.feature_expectations_addable(world, combined_trajs, feat_count, new_number)
		init_count = I.inital_probabilities_addable(world, combined_trajs, init_count, new_number)
		return combined_trajs, feat_count, init_count

	def parallel(self, n_run):
		delta = 0.05
		solver = deepcopy(self.solver)
		world = deepcopy(self.world)
		feat_count = np.zeros(world.state_space)
		init_count = np.zeros(world.state_space)
		reward = world.rewards
		old_trajs = []

		prelim = 10

		old_trajs = []
		feat_count = np.zeros((world.state_space))
		init_count = np.zeros((world.state_space))
		irl_array = np.zeros((world.state_space))
		episodes = 500
		prelim = 10
		reward = world.rewards
		delta = 0.05

		# print(solver.q_array)
		for ep in range(episodes-prelim):
			if ep == 0:
				combined_trajs, feat_count, init_count = self.add_trajectories(solver, old_trajs, world, 10000, 
					feat_count, init_count, False, reward)
				# print("como estas?")
				feature_expectation = feat_count/len(combined_trajs)
				p_initial = init_count/len(combined_trajs)
				reward, policy = I.m_irl(world, feature_expectation, p_initial, 200, 0.1)
			count = 0
			maxi = 0
			for t in combined_trajs:
				indi = 0
				for s in t.states():
					count+=1
					indi += 1
				if indi > maxi:
					maxi_traj = t
					maxi = indi
			print(solver.return_policy())
			print(count/10000)
			print(maxi)
			print(maxi_traj)
			exit()
			combined_trajs, feat_count, init_count = self.add_trajectories(solver, combined_trajs, world, 1, 
				feat_count, init_count, False, reward)
			feature_expectation = feat_count/len(combined_trajs)
			p_initial = init_count/len(combined_trajs)
			reward, policy = I.m_irl(world, feature_expectation, p_initial, 200, 0.1)

			old_trajs = combined_trajs
			# Normalize
			reward = reward/np.max(reward)
			if ep + prelim> 30 and reward[world.state_space-1]*(1-world.dif) - delta < reward[world.state_space - world.size] and reward[world.state_space - world.size] < reward[world.state_space-1]*(1-world.dif) + delta:
				# print("Success")
				return [n_run, 1, ep+prelim, reward, np.linalg.norm((solver.return_policy() - policy)), policy]
			elif ep + prelim==489:
				# print("Fail")
				return [n_run, 0, ep+prelim, reward, np.linalg.norm((solver.return_policy() - policy)), policy]

		