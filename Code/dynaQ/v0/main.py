import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import gridworld as W
import agents as A
import trajectory as T
import plot as P
import irl as I
import maxent as M
import pickle
import os
from copy import deepcopy
# Create gridworld function
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

def create_gw(full_size, p_slip, r_dif):
	# Size is going to be large NxN so make the middle MxM the solution task area
	# such that M << N
	return W.GridWorld(full_size, p_slip, r_dif)

def generate_policy(world, temp, discount, lr, planning_steps):
	solver = A.BoltzmannAgent(world, temp, discount, lr, planning_steps)
	solver.epsilon =1.0
	solver.train(200, None)
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

def reflexive_run(world, solver, t_to_add, irl_output, old_trajectories):
	t_list = add_trajectories(world, solver, t_to_add, irl_output, old_trajectories)
	rewards = I.m_irl(world, t_list, 200, 0.1)
	return rewards, t_list

def main():
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}

	difs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

	for r_dif in difs:
		# Create MDP
		world = create_gw(5,0.0,r_dif)

		# Create agent and solve
		agent = generate_policy(world, 0.05, 0.9, 0.1, 100)

		# Reflect my boy
		demos = 50
		averaging = 200
		error = 0.01
		epochs = 300000

		successes = 0

		listed = []

		tot_suc = 0
		for _ in tqdm(range(averaging)):

			solver = deepcopy(agent)

			# Generate a single trajectory
			t_list = add_trajectories(world, solver, 1)

			# Do IRL
			rewards = I.m_irl(world, t_list, epochs, 0.1)
			l_dif = world.rewards - rewards
			# print(e)

			best_e = np.inf
			best_reward = None
			best_pol = None
			best_demo = None

			for i in range(demos):
				# print(t_list)
				rewards, t_list = reflexive_run(world, solver, 1, l_dif, t_list)

				l_dif = world.rewards - rewards
				e = np.sum(np.square(l_dif))
				print(rewards.reshape((5,5)))
				print(l_dif.reshape((5,5)))
				print(e)

				if e<best_e:
					best_e = e
					best_reward = rewards
					best_pol = solver.return_policy()
					best_demo = i

				if e < 0.01:
					print("YAY")
					exit()
					successes += 1

			listed.append([successes, best_demo, best_e, best_reward, best_pol])

			if successes > 0:
				tot_suc += 1

		print(f"R_dif: {r_dif}, successful_iters_%: {100*(tot_suc/averaging)}")

		path = os.getcwd()+"/data/planning_step_100/r_dif/"+str(r_dif)+"/"

		with open(path+'100runs.pkl', 'wb') as filehandle:
			pickle.dump(listed, filehandle)

if __name__ == '__main__':
	main()