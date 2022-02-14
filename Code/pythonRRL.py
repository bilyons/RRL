import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

Sx = 21
Sy = Sx
S = (Sx*Sy)
A = 4
T = 10000000
TM= 10000000


class World:
	def __init__(self, height, width):
		self.env = np.zeros((height, width))
		self.height = height
		self.width = width

	def display(self, agent):
		self.env[agent.old_x][agent.old_y] = 0
		self.env[agent.x][agent.y] = 1
		print(self.env)

class Agent:
	def __init__(self, world, action_space, epsilon, eta, gamma):
		self.world = world;
		self.action_space = action_space;
		self.epsilon = epsilon;
		self.eta = eta;
		self.gamma = gamma;

		self.QArray = np.random.rand(world.height, world.width, action_space);
		self.VArray = np.random.rand(world.height, world.width);

		self.x = np.random.randint(0, world.height)
		self.y = np.random.randint(0, world.width)
		self.old_x = 0
		self.old_y = 0

	def get_reward(self):
		if self.x == self.old_x and self.y == self.old_y:
			self.r = -1.0
		else:
			self.r = 0.0

	def get_action(self):
		if np.random.rand()<self.epsilon:
			self.action = np.random.randint(0,4)
		else:
			self.action = np.argmax(self.QArray[self.x][self.y][:])
		return self.action

	def move(self, action):
		self.old_x = self.x
		self.old_y = self.y
		if action == 0: # Up
			if (self.x>0):
				self.x -= 1
		if action == 1: # Right
			if (self.y < (self.world.width-1)):
				self.y += 1

		if action == 2:
			if (self.x<(self.world.height - 1)):
				self.x += 1

		if action == 3:
			if (self.y>0):
				self.y - 1

	def updateValues(self):
		V = np.amax(self.QArray[self.x][self.y][:])
		Q = self.QArray[self.old_x][self.old_y][self.action]
		self.VArray[self.x][self.y] = V
		self.QArray[self.old_x][self.old_y][self.action] += self.eta*(self.r + self.gamma*V - Q)

def main():
	e = 0
	t = 0
	T = 12
	E = 10000000
	world=World(21,21)
	agent1 = Agent(world, A, 0.75, 0.10, 0.9)

	while e<E:
		t=0
		agent1.x, agent1.y = np.random.randint(0, 21), np.random.randint(0, 21)
		while t<T:
			a= agent1.get_action()
			agent1.move(a)
			agent1.get_reward()
			agent1.updateValues()
			t+=1
		e+= 1

	nx = 21
	ny = 21

	x = range(nx)
	y = range(ny)

	data = agent1.VArray

	hf = plt.figure()
	ha = hf.add_subplot(111, projection='3d')

	X,Y = np.meshgrid(x,y)
	ha.plot_surface(X,Y,data)

	plt.show()
	
if __name__ == '__main__':
	main()
