import matplotlib 
import numpy as np
import trajectory as T
import irl as I
import gridworld as W
from tqdm import tqdm
from copy import deepcopy
from time import sleep
class Parallel:

	def __init__(self, world, solver, demos, prelim, error):
		self.world = world
		self.solver = solver
		self.demos = demos
		self.prelim = prelim
		self.error = error

	def add_trajectories(self, world, solver, t_to_add, irl_output=None, old_trajectories=None):
		solver.epsilon = 0.0
		if old_trajectories is None:
			trajectories_list = []
		else:
			trajectories_list = old_trajectories
		for _ in range(t_to_add):
			trajectories_list.append(solver.run(irl_output))
		return trajectories_list

	def reflexive_run(self, world, solver, t_to_add, irl_output, old_trajectories):
		t_list = self.add_trajectories(world, solver, t_to_add, irl_output, old_trajectories)
		rewards = I.m_irl(world, t_list, 0.1)
		return rewards, t_list

	def parallel(self, n_run):
		delta = 0.05
		solver = deepcopy(self.solver)
		world = deepcopy(self.world)
		prelim = deepcopy(self.prelim)
		# print("yo")
		t_list = self.add_trajectories(world, solver, prelim)
		# print("it here?")
		rewards = I.m_irl(world, t_list, 0.1)
		# print("nah")
		r_dif = world.rewards - rewards

		successes = 0
		best_e = np.inf
		best_reward = None
		best_pol = None
		best_demo = None

		for i in range(self.demos):

			# print("guess not")

			rewards, t_list = self.reflexive_run(world, solver, 1, r_dif, t_list)
			# print("it's here")
			r_dif = world.rewards - rewards

			e = np.sum(np.square(r_dif))

			if e<best_e:
				best_e = e
				best_reward = rewards
				best_pol = solver.return_policy()
				best_demo = i
			if e < 0.01:
				successes += 1
				# print(f"{n_run} was successful")
				# break
		return [successes, best_demo, best_e, best_reward, best_pol]


		