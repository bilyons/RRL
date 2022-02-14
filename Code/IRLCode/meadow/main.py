"""
Main file for Meadow variant of RRL-IRL

Billy Lyons, 2021
billy.lyons@ed.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
import gridworld as W
import solvers as S
import maxent as I
import parallel as P
from parfor import pmap
from copy import deepcopy
import pickle
import os
# Set 5 decimal places for print checks
np.set_printoptions(formatter={'float': lambda x: "{0:0.20f}".format(x)})

# Create gridworld function

def create_gw(full_size, p_slip, r_dif, spawn_size):
	# Size is going to be large NxN so make the middle MxM the solution task area
	# such that M << N
	return W.GridWorld(full_size, p_slip, r_dif, spawn_size)

def create_agent(world, temp, t_length, discount, lr, t_eps):
	solver = S.BoltzmannAgent(world, t_length, temp, discount, lr)
	solver.train(t_eps)
	return solver

def add_trajectories(world, solver, t_to_add, rr=None, old_trajectories=None):
	if old_trajectories is None:
		trajectories_list = []
	else:
		trajectories_list = old_trajectories
	for t in range(t_to_add):
		trajectories_list.append(solver.run(rr))
	return trajectories_list

def reflexive_run(world, solver, t_to_add, rr, old_trajectories):
	t_list = add_trajectories(world, solver, t_to_add, rr, old_trajectories)
	rewards = I.irl(world, 0.8, t_list, 20, 0.1)
	return rewards, t_list

def main():



	s = 11
	spawns = [5,7,9]
	difs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

	for spawn in spawns:
		for r_dif in difs:
	# spawn = 5
	# r_dif = 0.5

			world = create_gw(s, 0.2, r_dif, spawn)
			expert = create_agent(world, 0.1, 20, 0.7, 0.1, 5000)

			expert.epsilon = 0.1

			# for s in range(world.n_states):
			# 	for a in range(world.n_actions):
			# 		if np.sum(world.transition_prob[s,:,a]) !=1:
			# 			print(s, a)
			# 			print(np.sum(world.transition_prob[s,:,a]) )
			# 			print(world.transition_prob[s,:,a])
			# exit()
			world_s = 'gridworld'
			size_s = str(spawn)

			check = deepcopy(expert)
			episodes = 200
			runs = 1
			delta = 0.05
			prelim = 1
			epochs = 20
			lr = 0.1

			paralleled = P.Parallel(world, check, episodes, delta, prelim, epochs, lr)
			listed = pmap(paralleled.parallel, range(runs))
			success = 0
			fail = 0
			for a in listed:
				if a[1] == 1:
					success+=1
				else:
					fail += 1
			listed.append(check.q_array)

			print(f"At {world_s}, size: {spawn}, reward diff: {r_dif}, Successes: {success}, Failures: {fail}")
			# print(listed)
			path = os.getcwd()+"/data/"+world_s+"/"+size_s+"/"
			# open(path+'run.pkl', 'w')
			with open(path+'{}.pkl'.format(r_dif), 'wb') as filehandle:
				pickle.dump(listed, filehandle)

if __name__ == '__main__':
	main()