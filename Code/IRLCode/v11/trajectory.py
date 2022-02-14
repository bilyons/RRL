import numpy as np
import random
from itertools import chain

class Trajectory:
	# Tales in a list of tuples (state, action, new state) and concerts it to the trajectory class
	# Should be easier to store multiple this way

	def __init__(self, transitions):
		self._t = transitions

	def transitions(self):

		return self._t

	def __repr__(self):
		return "Trajectory({})".format(repr(self._t))

	def __str__(self):
		return "{}".format(self._t)

	def states(self):
		return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

def add_trajectories(trajectories_list, trajectory):
	return trajectories_list.append(trajectory)
