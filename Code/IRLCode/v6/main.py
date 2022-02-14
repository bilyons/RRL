import matplotlib 
matplotlib.use('Agg') 
import numpy as np
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap
import gridworld as W
import solvers as S
import optimizer as O
import trajectory as T
import maxent as M
import parallel as L

import pickle
import os


def create_lw(size, slip, r_dif):
	# Rewards
	rewards = np.zeros(size)
	rewards[size-1] = 1.0
	rewards[0] = 1.0-r_dif
	# Terminal states
	terminals = [0, (size-1)]

	world = W.LineWorld(size, slip, terminals, rewards)
	return world, rewards, terminals

def create_gw(size, slip, r_dif):
	# Will need to function at larger than 5
	# Rewards
	rewards = np.zeros(size**2)
	rewards[(size**2-1)] = 1.0
	rewards[(size**2-size)] = 1.0-r_dif
	# Terminal states
	terminals = [(size**2-1), (size**2-size)]#, 0, size-1]
	world = W.GridWorld(size, slip, terminals, rewards)
	return world, rewards, terminals	

def generate_policy(world, rewards, terminals):
	solver = S.BoltzmannAgent(world, 0.1, 0.90, 0.1)
	solver, policy, v = S.gen_policy(world, rewards, terminals, solver, 5000)
	return solver, policy, v

def maxent(world, terminals, trajectories, feat_count, init_count, ep_number):
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
	reward, feat_count, init_count = M.irl(world, features, terminals, trajectories, 
		optim, init, feat_count, init_count, ep_number)

	return reward, feat_count, init_count

def main():
	sizes = [7,9]
	difs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
	for size in sizes:
		for r_dif in difs:
			# world, rewards, terminals = create_lw(size, 0.2, r_dif)
			world, rewards, terminals = create_gw(size, 0.2, r_dif)
			world_s = 'gridworld'
			size_s = str(size)
			# 
			# Generate expert policy
			solver, policy, v = generate_policy(world, rewards, terminals)

			# save solver base policy
			parallelo = L.Parallel(world, solver, 500)

			# RRL Section
			runs = 500
			episodes = 500
			delta = 0.05
			non_rrl = []
			rrl = []
			listed = pmap(parallelo.parallel, range(runs))

			path = os.getcwd()+"/data/"+world_s+"/"+size_s+"/"
			# open(path+'run.pkl', 'w')
			with open(path+'{}.pkl'.format(r_dif*10), 'wb') as filehandle:
				pickle.dump(listed, filehandle)

if __name__ == '__main__':
	main()