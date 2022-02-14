import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import gridworld as W
import agents as A
import trajectory as T
import plot as P
import irl as I

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

def reflexive_run(world, solver, t_to_add, irl_output, old_trajectories):
	t_list = add_trajectories(world, solver, t_to_add, irl_output, old_trajectories)
	rewards = I.m_irl(world, t_list, 0.1)
	return rewards, t_list

def main():
	style = {
		'border': {'color': 'red', 'linewidth': 0.5},
	}
	# Create MDP
	world = create_gw(5,0.0,0.5)

	# Create agent and solve
	agent = generate_policy(world, 0.05, 0.9, 0.1, 100)

	# Generate a single trajectory
	t_list = add_trajectories(world, agent, 2)

	# Do IRL
	rewards = I.m_irl(world, t_list, 0.1)

	r_dif = world.rewards - rewards

	# exit()
	ax = plt.figure(num='After training').add_subplot(111)
	P.plot_stochastic_policy(ax, world, agent.return_policy(), **style)
	# print(rewards.reshape((5,5)))
	plt.draw()
	for i in tqdm(range(50)):
		rewards, t_list = reflexive_run(world, agent, 1, r_dif, t_list)

		r_dif = world.rewards - rewards
		e = np.sum(np.square(r_dif).reshape((5,5)))
		# r_dif = (r_dif>0)*r_dif
		print(r_dif.reshape((5,5)))
		if e < 0.01:
			ax = plt.figure(num='Policy after {} reflections'.format(i)).add_subplot(111)
			P.plot_stochastic_policy(ax, world, agent.return_policy(), **style)
			plt.draw()
			ax = plt.figure(num='Reward at {} reflections'.format(i)).add_subplot(111)
			P.plot_state_values(ax, world, rewards, **style)
			plt.draw()
		# plt.show()

		# print(f"Reward after reflection {i}\n")
		# print(rewards.reshape(5,5))
		# print("squared error\n")
		# print(np.square(r_dif).reshape((5,5)))
		# print(np.sum(np.square(r_dif).reshape((5,5))))
		# print(r_dif.reshape((5,5)))
	ax = plt.figure(num='Policy after {} reflections'.format(i)).add_subplot(111)
	P.plot_stochastic_policy(ax, world, agent.return_policy(), **style)
	plt.draw()
	ax = plt.figure(num='Reward at {} reflections'.format(i)).add_subplot(111)
	P.plot_state_values(ax, world, rewards, **style)
	plt.draw()
	plt.show()
	# rewards, t_list = reflexive_run(world, agent, 1, r_dif, t_list)

	# print(rewards.reshape((5,5)))
	# print(r_dif.reshape((5,5)))


	# t_list = add_trajectories(world, agent, 1, r_dif, t_list)
	# fe = I.feature_expectations_addable(world, t_list, feat_count, 1)
	# p_i = I.inital_probabilities_addable(world, t_list, init_count, 1)

	# rewards = I.m_irl(world, fe, p_i, 400, 0.1, None)
	# r_dif = world.rewards - rewards
	# print(rewards.reshape((5,5)))
	# print(r_dif.reshape((5,5)))


	plt.show()
	# print(rewards.reshape((5,5)))
if __name__ == '__main__':
	main()