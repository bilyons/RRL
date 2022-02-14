import numpy as np
import gridworld as W
import solvers as S
import trajectory as T
import optimizer as O
import maxent as M
import plot as P
import matplotlib.pyplot as plt

start_epsilon_decay =1
end_epsilon_decay = 0.1

def create_world():
	world = W.GridWorld(5)
	# Rewards
	rewards = np.zeros(world.state_space)
	rewards[24] = 1.0
	rewards[20] = 1.0
	# Terminal states
	terminals = [20, 24]

	return world, rewards, terminals

def generate_policy(world, rewards, terminals):

	n_trajectories = 2
	temp = 0.1
	discount = 0.65
	lr = 0.2

	# Solve the MDP ############################
	solver = S.BoltzmannAgent(world, temp, discount)
	#solver = S.GreedyAgent(world, temp, discount, lr)
	# for a in range(len(terminals)):
	# 	terminal = terminals[a]
	# 	for b in range(world.action_space):
	# 		solver.q_array[terminal][b] = 1.0
	episodes = 100000
	e_decay_value = (start_epsilon_decay - end_epsilon_decay)/episodes
	eps = 0
	while eps < episodes:
		done = False
		t=0
		# Generates first state, not inclusive of end
		s = np.random.randint(0,24)
		while s in terminals:
			s = np.random.randint(0,24)
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
			if new_s in terminals:
				done = True
				#print(solver.epsilon, eps)
			s = new_s
			old_state = new_state
			t+=1
		eps+= 1
		solver.epsilon_decay(e_decay_value)

	#print(solver.epsilon)
	v = np.zeros(25)
	for x in range(25):
		v[x] = np.max(solver.q_array[x])

	policy = np.zeros((world.state_space, world.action_space))
	for s in range(world.state_space):
		prob = np.exp(np.divide(solver.q_array[s], solver.temp))
		sum_prob = np.sum(prob)
		dist = np.divide(prob, sum_prob)
		policy[s] = dist

	return policy	

def generate_trajectories(n_trajectories, policy, world, rewards, terminals):
	trajs = list(T.generate_trajectories(n_trajectories, world, policy, 1, terminals))
	return trajs

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
    expert_policy = generate_policy(world, rewards, terminals)

    # generate first trajectory
    trajectories = generate_trajectories(1, expert_policy, world, rewards, terminals)

    # get predicted reward scheme from trajectory
    reward_maxent = maxent(world, terminals, trajectories)

    print(np.reshape(reward_maxent, (5,5)))

    for a in range(4):
    	new_trajectory = generate_trajectories(1, expert_policy, world, rewards, terminals)
    	trajectories = add_trajectory(trajectories, new_trajectory)
    	reward_maxent = maxent(world, terminals, trajectories)
    	print(np.reshape(reward_maxent, (5,5)))


    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    P.plot_stochastic_policy(ax, world, expert_policy, **style)

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