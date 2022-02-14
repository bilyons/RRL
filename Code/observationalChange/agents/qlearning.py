import numpy as np
import pandas as pd
import itertools

def q_update(q_array, state, action, new_state, reward, discount, lr):
	q_copy = q_array.copy()
	# print(q_array)
	old_q = q_copy[state, action]
	max_future_q = np.max(q_copy[new_state,:])
	delta = lr*(reward + discount*max_future_q - old_q)
	# print(delta)
	q_copy[state, action]+=delta
	# print(q_copy)
	# exit()
	return q_copy

def action_selection(q_array, state, tau):
	exponentiated = np.exp(q_array[state,:]/tau)
	prob = exponentiated/np.sum(exponentiated)
	action = np.random.choice(len(q_array.T), p=prob)
	return action