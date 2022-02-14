import numpy as np

class Storage(object):

	def __init__(self, world, rewards, terminals, expert_pol, max_length):
		self.world = world
		self.rewards = rewards
		self.terminals = terminals
		self.expert_pol = expert_pol
		self.max_length = max_length