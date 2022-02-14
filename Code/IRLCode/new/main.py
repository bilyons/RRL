import numpy as np
import gridworld as W
import solvers as S
import plot as P
import value_iteration as V
import maxent as I

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
# import irl as I

import matplotlib.pyplot as plt

def create_gw(size, p_slip, r_dif, discount):
	rewards = np.zeros(size**2)
	rewards[size**2-1] = 1.0
	# rewards[size**2 - size] = 1.0 - r_dif

	return W.GridWorld(size, rewards, p_slip, discount)

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

	# Create world
	world = create_gw(5, 0.2, 0.5, 0.8)
	print(V.value_iteration(world, world.transition_prob, world.rewards, 0.5, 0.01))
	exit()
	# Create agent - solve MDP
	expert = create_agent(world, 0.5, 20, 0.8, 0.1, 5000)

	t_list = add_trajectories(world, expert, 10)

	rewards = I.irl(world, 0.8, t_list, 30, 0.1)
	delta = 0.05
	for i in range(500):
		print(i)
		rewards, t_list = reflexive_run(world, expert, 1, rewards, t_list)
		print(rewards.reshape((5,5)))
		if rewards[world.n_states-1]*(1-0.5) - delta < rewards[world.n_states - world.size] and rewards[world.n_states - world.size] < rewards[world.n_states-1]*(1-0.5) + delta:
			print("Success")
			
	# Plot policy
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}


if __name__ == "__main__":
	main()