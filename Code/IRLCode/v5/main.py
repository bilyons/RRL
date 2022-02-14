import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap
from multiprocessing import Pool

import gridworld as W
import solvers as S
import plot as P
import optimizer as O
import trajectory as T
import maxent as M
import parallel as I

def create_lw(size, slip):
	# Rewards
	rewards = np.zeros(size)
	rewards[size-1] = 1.0
	rewards[0] = 0.65
	# Terminal states
	terminals = [0, (size-1)]

	world = W.LineWorld(size, slip, terminals, rewards)
	return world, rewards, terminals

def create_gw(size, slip):
	# Will need to function at larger than 5
	# Rewards
	rewards = np.zeros(size**2)
	rewards[(size**2-1)] = 1.0
	rewards[(size**2-size)] = 0.65
	# Terminal states
	terminals = [(size**2-1), (size**2-size)]
	world = W.GridWorld(size, slip, terminals, rewards)
	return world, rewards, terminals	

def generate_policy(world, rewards, terminals):
	solver = S.BoltzmannAgent(world, 0.1, 0.90, 0.1)
	solver, policy, v = S.gen_policy(world, rewards, terminals, solver, 5000)
	return solver, policy, v

def main():
	# set-up mdp
	world, rewards, terminals = create_lw(7, 0.2)
	# world, rewards, terminals = create_gw(5, 0.2)
	# Generate expert policy
	solver, policy, v = generate_policy(world, rewards, terminals)
	# save solver base policy
	base_pol = np.copy(solver.q_array)
	# info class


	# RRL Section
	runs = 100
	episodes = 500
	delta = 0.05

	info = I.Parallel(world, rewards, terminals, base_pol, episodes)
	# pool = Pool()
	# print(range(runs))
	# exit()
	for i in range(100):
		info.paralleled(i)

	exit()
	print(pmap(info.paralleled, range(runs)))


	# print("RRL within acceptable margin percentage: ", (sum(rrl)/50)*100)
	# print("Primal IRL within acceptable margin percentage: ", sum(non_rrl))
	# show our original reward
	# ax = plt.figure(num='Original Reward').add_subplot(111)
	# P.plot_state_values(ax, world, rewards, **style)
	# plt.draw()

	# # show our expert policies
	# ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
	# P.plot_stochastic_policy(ax, world, solver.return_policy(), **style)

	# plt.draw()

	# # show the computed reward
	# ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
	# P.plot_state_values(ax, world, reward_maxent, **style)
	# plt.draw()

	# show the computed reward
	# ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
	# P.plot_state_values(ax, world, reward_causal, **style)
	# plt.draw()
	
	# plt.show()


if __name__ == '__main__':
	main()