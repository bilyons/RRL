import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import gridworld as W
import agents as A
import trajectory as T
import plot as P
import irl as I
import parallel as L
from parfor import pmap
import pickle
import os
# Create gridworld function
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

def create_gw(full_size, p_slip, r_dif):
	# Size is going to be large NxN so make the middle MxM the solution task area
	# such that M << N
	return W.GridWorld(full_size, p_slip, r_dif)

def generate_policy(world, temp, discount, lr, planning_steps):
	solver = A.BoltzmannAgent(world, temp, discount, lr, planning_steps)
	solver.epsilon =1.0
	solver.train(100, None)
	return solver

def add_trajectories(world, solver, t_to_add, irl_output=None, old_trajectories=None):
	solver.epsilon = 0.0
	if old_trajectories is None:
		trajectories_list = []
	else:
		trajectories_list = old_trajectories
	for _ in range(t_to_add):
		trajectories_list.append(solver.run(irl_output))
	return trajectories_list

def reflexive_run(world, solver, t_to_add, irl_output, old_trajectories):
	t_list = add_trajectories(world, solver, t_to_add, irl_output, old_trajectories)
	rewards = I.m_irl(world, t_list, 0.1)
	return rewards, t_list

def main():
	# Cycle through number of planning steps
	# for i in range(1, 101):

	# 	# Cycle through number of trajectories presampled
	# 	for j in range(1, 11):

	# Cycle through reward difference
	difs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
	for r_dif in difs:
		# Create MDP
		world = create_gw(5,0.0,r_dif)

		# Create agent and solve
		agent = generate_policy(world, 0.05, 0.9, 0.1, 50)

		# Parallel

		demos = 50
		prelim = 1
		error = 0.01
		averaging = 200

		t_list = add_trajectories(world, agent, prelim)

		rewards = I.m_irl(world, t_list, 0.1)

		l_dif = world.rewards - rewards

		for i in tqdm(range(demos)):
			rewards, t_list = reflexive_run(world, agent, 1, l_dif, t_list)

			e = np.sum(np.square(l_dif))

			if e < 0.01:
				print("success")
				print(rewards.reshape((5,5)))

		path = os.getcwd()+"/data/planning_step_100/r_dif/"+str(r_dif)+"/"

		with open(path+'100runs.pkl', 'wb') as filehandle:
			pickle.dump(listed, filehandle)

if __name__ == '__main__':
	main()