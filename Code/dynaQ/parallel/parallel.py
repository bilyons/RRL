import matplotlib 
import numpy as np

import irl as I
import gridworld as W
import agents as A
import trajectory as T

from tqdm import tqdm
from copy import deepcopy
from time import sleep

class Parallel:

	def __init__(self, full_size, p_slip, r_dif, prelim, error, run_length):
		self.full_size = full_size
		self.p_slip = p_slip
		self.r_dif = r_dif
		self.prelim = prelim
		self.error = error
		self.run_length = run_length

	def create_gw(self, full_size, p_slip, r_dif):
		# Size is going to be large NxN so make the middle MxM the solution task area
		# such that M << N
		return W.GridWorld(full_size, p_slip, r_dif)

	def generate_policy(self, world, temp, discount, lr, planning_steps):
		solver = A.BoltzmannAgent(world, temp, discount, lr, planning_steps)
		solver.epsilon =1.0
		solver.train(100, None)
		return solver

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
		rewards = I.m_irl(world, t_list, 20000, 0.1)
		return rewards, t_list

	def parallel(self, n_run):
		world = self.create_gw(self.full_size, self.p_slip, self.r_dif)
		solver = self.generate_policy(world, 0.05, 0.9, 0.1, 50)
		t_list = self.add_trajectories(world, solver, self.prelim)

		rewards = I.m_irl(world, t_list, 20000, 0.1)

		r_dif = world.rewards - rewards

		successes = 0
		best_e = np.inf
		best_reward = None
		best_pol = None
		best_demo = None

		for i in range(self.run_length-self.prelim):

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
			if e < self.error:
				successes += 1
				# print(f"{n_run} was successful")
				# break
		return [successes, best_demo, best_e, best_reward, best_pol]


		