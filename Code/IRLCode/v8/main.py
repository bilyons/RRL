import numpy as np
import gridworld as W
import solvers as S
import trajectory as T
import irl as I

from tqdm import tqdm
from time import sleep

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

def create_lw(size, p_slip, r_dif):
	# Creates a 1D world of length size
	# Rewards
	rewards = np.zeros(size)
	rewards[size-1] = 1.0
	rewards[0] = 1.0 - r_dif
	# Terminals
	terminals = [0, (size-1)]

	world = W.LineWorld(size, p_slip, terminals, rewards)
	return world

def create_gw(size, p_slip, r_dif):
	# Creates a grid world size x size
	# Rewards
	rewards = np.zeros(size**2)
	rewards[(size**2-1)] = 1.0
	rewards[(size**2-size)] = 1.0-r_dif
	# Terminal States
	terminals = [(size**2-1), (size**2-size)]#, 0, size-1]
	world = W.GridWorld(size, p_slip, terminals, rewards)
	return world

def generate_policy(world, temp, discount, lr, t_eps):
	solver = S.BoltzmannAgent(world, temp, discount, lr)
	solver.train(t_eps)
	# print(solver.return_policy())
	return solver

def add_trajectories(solver, old_trajs, world, new_number, feat_count, init_count, r_bool, irl_array):
	new_trajs = RRL(solver, world, new_number, r_bool, irl_array)
	combined_trajs = old_trajs + new_trajs
	feat_count = I.feature_expectations_addable(world, combined_trajs, feat_count, new_number)
	init_count = I.inital_probabilities_addable(world, combined_trajs, init_count, new_number)
	return combined_trajs, feat_count, init_count


def RRL(solver, world, num_trajectories, r_bool, irl_array):
	# Takes in the solver, world, and a desired number of trajectories to add in the batch
	# Outputs the new trajectories

	trajectories = []
	# For number of trajectories to add
	for t in range(num_trajectories):
		# Initialise empty trajectory
		trajectory = []
		# Set done to false for solve
		done = False
		# Initialise
		s = np.random.randint(0, world.state_space)
		while s in world.terminals: # Catches initialising in terminal. Not allowed
			s = np.random.randint(0, world.state_space) 

		# Episode
		while not done:
			# Draw action
			action = solver.action_selection(s)
			# Get new state
			new_s = world.movement(s, action)

			# Get reward
			reward = world.rewards[new_s]

			if new_s in world.terminals:
				done = True
				if r_bool:
					reward += world.reflexive_reward(new_s, irl_array, world.rewards)

			solver.update_values(s, action, reward, new_s)
			trajectory += [(s, action, new_s)]

			s = new_s

		trajectory = T.Trajectory(trajectory)

		trajectories.append(trajectory)

	return trajectories


def main():

	sizes = [5,7,9]
	difs = [0.0,0.1,0.2,0.3,0.4,0.5]
	
	for size in sizes:
		for r_dif in difs:
		# Set up the MDP - generate the world
			world = create_gw(size, 0.2, r_dif)

			# Generate the expert and solve the MDP
			solver = generate_policy(world, 0.1, 0.9, 0.1, 5000)

			expert_pol = solver.q_array

			world_s = 'gridworld'
			size_s = str(size)

			output = []
			for run in range(500):
				
				solver.q_array = expert_pol
				# Initial counts
				feat_count = np.zeros((world.state_space))
				init_count = np.zeros((world.state_space))

				solver.epsilon = 0

				# Reflexive Reinforcement Learning
				# I want to be able to add multiple runs at once

				old_trajs = []
				reward = np.random.rand(world.state_space)
				episodes = 500
				delta = 0.05
				for ep in tqdm(range(episodes)):
					combined_trajs, feat_count, init_count = add_trajectories(solver, old_trajs, world, 1, 
						feat_count, init_count, True, reward)

					feature_expectation = feat_count/len(combined_trajs)
					p_initial = init_count/len(combined_trajs)
					reward = I.m_irl(world, feature_expectation, p_initial, 200, 0.1, reward)
					old_trajs = combined_trajs

					# Normalize
					reward = reward/np.max(reward)
					if ep > 50 and reward[world.state_space-1]*0.65 - delta < reward[world.state_space-world.size] and reward[world.state_space-world.size] < reward[world.state_space-1]*0.65 + delta:
						output.append([run, 1, ep, reward])
						break
					elif ep == 499:
						output.append([run, 0, ep, reward])

			path = os.getcwd()+"/data/"+world_s+"/"+size_s+"/"
			# open(path+'run.pkl', 'w')
			with open(path+'{}.pkl'.format(r_dif*10), 'wb') as filehandle:
				pickle.dump(output, filehandle)

if __name__ == '__main__':
	main()