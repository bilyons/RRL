import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
from tqdm import tqdm

import gridworld as W
import solvers as S
import plot as P
import optimizer as O
import trajectory as T
import maxent as M

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

def RRL_run(solver, world, rewards, terminals, maxent):
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
		# eff_a = world.effective_action(action, s, new_s)
		# action = eff_a

		if new_s in terminals:
			done = True
			reward += world.reflexive_reward(new_s, maxent, rewards)
			# print(reward, new_s)
			# print(reward)
		# print(reflexive[1], action, reward)

		# update values
		solver.update_values(s, action, reward, new_s)
		trajectory += [(s, action, new_s)]
		
			#print(solver.epsilon, eps)
		s = new_s
	# print(len(trajectory))
	return T.Trajectory(trajectory)

def add_trajectory(trajectories, new_trajectory):
	return T.add_trajectories(trajectories, new_trajectory)

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

def maxent_causal(world, terminals, trajectories, feat_count, init_count, ep_number,  discount=0.8):
	"""
	Maximum Causal Entropy Inverse Reinforcement Learning
	"""
	# set up features: we use one feature vector per state
	features = W.state_features(world)

	# choose our parameter initialization strategy:
	#   initialize parameters with constant
	init = O.Constant(1.0)

	# choose our optimization strategy:
	#   we select exponentiated gradient descent with linear learning-rate decay
	optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

	# actually do some inverse reinforcement learning
	reward = M.irl_causal(world, features, terminals, trajectories, optim, init, 
		feat_count, init_count, ep_number, discount)

	return reward

def main():
	# common style arguments for plotting
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}

	# set-up mdp
	world, rewards, terminals = create_lw(7, 0.2)
	# world, rewards, terminals = create_gw(5, 0)
	# Generate expert policy
	solver, policy, v = generate_policy(world, rewards, terminals)

	# save solver base policy
	base_pol = np.copy(solver.q_array)

	# RRL Section
	runs = 500
	episodes = 500
	delta = 0.05
	non_rrl = []
	rrl = []
	for run in range(runs):

		if run>0:
			# Reload policy
			solver.q_array = np.copy(base_pol)

		feat_count = np.zeros(world.state_space)
		init_count = np.zeros(world.state_space)
		reward_maxent = rewards

		for ep in tqdm(range(episodes)):
			new_trajectory = (RRL_run(solver, world, rewards, terminals, reward_maxent))
			lengthot = 0
			# for x in new_trajectory.states():
			# 	lengthot +=1
			# if lengthot > 30:
			# 	print(solver.value_func())
			# print(solver.q_array)
			# exit()
			reward_maxent, feat_count, init_count = maxent(world, terminals, new_trajectory, 
				feat_count, init_count, ep)
			# reward_causal= maxent_causal(world, terminals, new_trajectory, 
			# 	feat_count, init_count, ep,discount=0.8)

			# if run%2 == 0:
			if ep>100 and reward_maxent[world.state_space-1]*0.65 - delta < reward_maxent[world.state_space-world.size] and reward_maxent[world.state_space-world.size] < reward_maxent[world.state_space-1]*0.65 + delta:
				# if run%2 == 0:
				rrl.append(1)
				reward_maxent = reward_maxent/reward_maxent[world.size-1]
				# else:
				# 	non_rrl.append(1)
				break

	print("RRL within acceptable margin percentage: ", (sum(rrl)/500)*100)
	print("Primal IRL within acceptable margin percentage: ", sum(non_rrl))
	# show our original reward
	ax = plt.figure(num='Original Reward').add_subplot(111)
	P.plot_state_values(ax, world, rewards, **style)
	plt.draw()

	# # show our expert policies
	# ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
	# P.plot_stochastic_policy(ax, world, solver.return_policy(), **style)

	# plt.draw()

	# # show the computed reward
	ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
	P.plot_state_values(ax, world, reward_maxent, **style)
	plt.draw()

	# show the computed reward
	# ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
	# P.plot_state_values(ax, world, reward_causal, **style)
	# plt.draw()
	
	plt.show()


if __name__ == '__main__':
	main()