import numpy as np
import gridworld as W
import solvers as S
import trajectory as T
import optimizer as O
import maxent as M
import plot as P
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import linalg as LA

start_epsilon_decay =1
end_epsilon_decay = 0.1

def create_world():
	world = W.GridWorld(5)
	# Rewards
	rewards = np.zeros(world.state_space)
	rewards[(world.size**2-1)] = 1.0
	rewards[(world.size**2-world.size)] = 0.65
	# Terminal states
	terminals = [(world.size**2-1), (world.size**2-world.size)]

	print(world._transition_prob_table())
	exit()

	return world, rewards, terminals

def generate_policy(world, rewards, terminals):
	temp = 0.5
	discount = 0.8
	lr = 0.01

	solver = S.BoltzmannAgent(world, temp, discount)

	policy = np.zeros((world.state_space, world.action_space))
	for s in range(world.state_space):
		prob = np.exp(np.divide(solver.q_array[s], solver.temp))
		sum_prob = np.sum(prob)
		dist = np.divide(prob, sum_prob)
		policy[s] = dist
	return solver

def generate_trajectories(n_trajectories, policy, world, rewards, terminals):
	trajs = list(T.generate_trajectories(n_trajectories, world, policy, 1, terminals))
	return trajs

def RRL_run(solver, world, rewards, terminals, episode):
	done = False
	trajectory = []
	s = np.random.randint(0, world.state_space)
	eta = 1
	while s in terminals:
		s = np.random.randint(0, world.state_space)
	old_state = world.state_to_coordinate(s)
	while not done:
		action = solver.action_selection(s)
		# convert to coord
		new_state = world.movement(old_state, action)
		# convert new coord
		new_s = world.coordinate_to_state(new_state)
		# get reward 
		reward = rewards[new_s]

		# update values
		solver.update_values(s, action, reward, new_s)
		trajectory += [(s, action, new_s)]
		if new_s in terminals:
			done = True
			#print(solver.epsilon, eps)
		s = new_s
		old_state = new_state
	return T.Trajectory(trajectory)

def add_trajectory(trajectories, new_trajectory):
	return T.add_trajectories(trajectories, new_trajectory)

def maxent(world, terminals, trajectories):
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
	reward = M.irl(world, features, terminals, trajectories, optim, init)

	return reward

#maxent(world, terminals, [])

def main():
	# common style arguments for plotting
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}

	# set-up mdp
	world, rewards, terminals = create_world()

	# show our original reward
	ax = plt.figure(num='Original Reward').add_subplot(111)
	P.plot_state_values(ax, world, rewards, **style)
	plt.draw()

	# generate "expert" policy
	solver = generate_policy(world, rewards, terminals)
	np.save("solver_q_array.npy", solver.q_array)

	# generate first trajectory from solver
	trajectories = generate_trajectories(1, solver.return_policy(), world, rewards, terminals)
	#print(trajectories)
	# get predicted reward scheme from trajectory
	reward_maxent = maxent(world, terminals, trajectories)

	#print(np.reshape(reward_maxent, (5,5)))

	episodes = 2000
	for ep in tqdm(range(episodes)):
		new_trajectory = (RRL_run(solver, world, rewards, terminals, ep))
		new_trajectory = [new_trajectory]
		trajectories = add_trajectory(trajectories, new_trajectory)
		reward_maxent = maxent(world, terminals, trajectories)

	ground_truth = rewards
	new = ground_truth - reward_maxent
	output = np.reshape(new, (world.size,world.size))

	frob = LA.norm(output, ord='fro')
	print(output)
	print(frob)

	# show our expert policies
	ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
	P.plot_stochastic_policy(ax, world, solver.return_policy(), **style)

	for t in trajectories:
		P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

	plt.draw()

	# maximum entropy reinforcement learning (non-causal)
	reward_maxent = maxent(world, terminals, trajectories)

	# show the computed reward
	ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
	P.plot_state_values(ax, world, reward_maxent, **style)
	plt.draw()

	plt.show()


if __name__ == '__main__':
	main()