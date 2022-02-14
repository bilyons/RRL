import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import gridworld as W
import agents as A
import trajectory as T
import plot as P
import irl as I
from copy import deepcopy
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
	for t in range(t_to_add):
		trajectories_list.append(solver.run(irl_output))
	return trajectories_list

def reflexive_run(world, solver, t_to_add, irl_output, old_trajectories, rewards):
	t_list = add_trajectories(world, solver, t_to_add, irl_output, old_trajectories)
	rewards = I.m_irl(world, t_list, 0.1)
	return rewards, t_list

def main():
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}

	# r_difs = [0.5,0.4,0.3,0.2,0.1,0.0]
	r_dif = 0.0
	# for r_dif in r_difs:
	# 	print(f"Reward difference: {r_dif}")
		# Create MDP
	world = create_gw(5,0.0, r_dif)

	# Create agent and solve
	agent = generate_policy(world, 0.05, 0.9, 0.1, 100)

	runs = 200
	print(f"Reward disparity: {r_dif}")
	for run in range(runs):
		solved = False
		i = 2

		# Copy trained agent
		copy = deepcopy(agent)

		# Add first two trajectories
		t_list = add_trajectories(world, copy, 2)

		# Do first IRL
		rewards = I.m_irl(world, t_list, 0.1)

		l_dif = world.rewards - rewards

		while not solved:

			# Get policy before success
			pol = agent.return_policy()

			# Perform RRL run
			rewards, t_list = reflexive_run(world, copy, 1, l_dif, t_list, rewards)

			l_dif = world.rewards - rewards

			e = np.sum(np.square(l_dif))

			i += 1

			if e<0.01:

				# Successful!
				solved = True

				print(f"Agent {run} successful at demo {i}")

				result = [i, rewards, e, pol]

				path = os.getcwd()+"/data/serial/r_dif/"+str(r_dif)+"/100/"

				with open(path+'run_{}.pkl'.format(run), 'wb') as filehandle:
					pickle.dump(result, filehandle)


		


if __name__ == '__main__':
	main()