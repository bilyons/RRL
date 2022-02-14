import numpy as np

world = np.zeros(25)
rewards = np.zeros(25)
terminals = [24]

for a in range(len(terminals)):
	end_state = terminals[a]
	rewards[end_state] = 1.0

state_space = len(world)
actions = [-1, 1]
action_space = len(actions)

q_array = np.zeros((state_space, action_space))

epsilon = 1
lr = 0.1
discount = 0.65

def move(state, action):
	movement = actions[action]
	new_state = state + movement
	if new_state < 0:
		new_state = 0
	return new_state

def action_selection(state):
	if np.random.rand() > epsilon:
		action = np.argmax(q_array[state])
	else:
		action = np.random.randint(0,2)
	return action

def update_values(old_state, action, new_state, reward):
	cur_q = q_array[old_state][action]
	max_future_q = np.max(q_array[new_state])
	new_q = cur_q + lr*(reward + discount*max_future_q - cur_q)
	q_array[old_state][action] = new_q

e = 0 
for e in range(100):
	done = False
	start_state = np.random.randint(0,25)
	while start_state in terminals:
		start_state = np.random.randint(0,25)

	while not done:
		action = action_selection(start_state)

		new_state = move(start_state, action)
		reward = rewards[new_state]
		update_values(start_state, action, new_state, reward)

		start_state = new_state

		if start_state in terminals:
			done = True

	e+=1
	epsilon *= 0.995

print(q_array)

value = np.zeros(25)
for a in range(25):
	value[a] = np.max(q_array[a])
print(value)
