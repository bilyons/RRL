import matplotlib 
import numpy as np
import trajectory as T
import maxent as I
import gridworld as W
from tqdm import tqdm
from copy import deepcopy
from time import sleep
class Parallel:

	def __init__(self, world, solver, episodes, delta, prelim, epochs, lr):
		self.world = world
		self.solver = solver
		self.episodes = episodes
		self.delta = delta
		self.prelim = prelim
		self.epochs = epochs
		self.lr = lr

	def add_trajectories(self,world, solver, t_to_add, rr=None, old_trajectories=None):
		if old_trajectories is None:
			trajectories_list = []
		else:
			trajectories_list = old_trajectories
		for t in range(t_to_add):
			trajectories_list.append(solver.run(rr))
		return trajectories_list

	def reflexive_run(self, world, solver, t_to_add, rr, old_trajectories):
		t_list = self.add_trajectories(world, solver, t_to_add, rr, old_trajectories)
		rewards = I.irl(world, self.solver.gamma, t_list, self.epochs, self.lr)
		return rewards, t_list

	def parallel(self, n_run):
		solver = deepcopy(self.solver)
		world = deepcopy(self.world)
		t_list = self.add_trajectories(world, solver, self.prelim)
		# print("yup")
		rewards= I.irl(world, self.solver.gamma, t_list, self.epochs, self.lr)
		for i in range(self.episodes-self.prelim):
			rewards, t_list = self.reflexive_run(world, solver, 1, rewards, t_list)
			# print((world.rewards - rewards).reshape((11,11)))
			# print("Error: ", np.sum(np.square(world.rewards - rewards)))

			# # print(world.rewards - rewards)
			# print(rewards[world.large_r]*(1-world.r_dif) - self.delta)
			# if rewards[world.large_r]*(1-world.r_dif) - self.delta < rewards[world.small_r] and rewards[world.small_r] < rewards[world.large_r]*(1-world.r_dif) + self.delta:
			# 	# print(n_run, 1, i+self.prelim, rewards, solver.return_policy())
			# 	return [n_run, 1, i+self.prelim, rewards, solver.return_policy()]
			# elif i + self.prelim==self.episodes-1:
			# 	return [n_run, 0, i+self.prelim, rewards, solver.return_policy()]

		