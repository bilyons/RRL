import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eps = 0.75
eta = 0.1
gamma = 0.9
alpha = 0.1


class Agent:
	def __init__(self, Sx, Sy, A):
		self.Sx = Sx
		self.Sy = Sy
		self.A = A
		self.Q = np.random.rand(Sx, Sy, A)
		self.V = np.random.rand(Sx, Sy)
		self.s0_x, self.s0_y = 0, 0
		self.s1_x, self.s1_y = 0, 0

	def reset(self):
		self.s0_x, self.s0_y = np.random.randint(0, (self.Sx)), np.random.randint(0, (self.Sy))
		self.s1_x, self.s1_y = self.s0_x, self.s0_y 
		return

	def get_action(self):
		if np.random.rand() < eps:
			action = np.random.randint(0,(self.A))
			#print("Random action {}".format(action))
		else:
			action = np.argmax(self.Q[self.s0_x][self.s0_y][:])
			#print("Maximal action {}".format(action))
		return action

	def move(self, action):
		self.s1_x, self.s1_y = self.s0_x, self.s0_y
		if action==0: # Left
			if (self.s0_x%self.Sx>0):
				self.s1_x = self.s0_x-1;
		elif action==1: # Down
			if (self.s0_y%self.Sy<self.Sy-1):
				self.s1_y = self.s0_y+1;
		elif action==2: # Right
			if (self.s0_x%self.Sx<self.Sx-1):
				self.s1_x = self.s0_x+1;
		elif action==3: # Up
			if (self.s0_y%self.Sy>0):
				self.s1_y = self.s0_y-1;
		else:
			print("Action chosen, {}, is impossible".format(action))
			exit(0)

	def reward(self):
		if (self.s1_x == self.s0_x) and (self.s1_y==self.s0_y):
			reward = -1
		else:
			reward = 0
		return reward

	def update_arrays(self, reward, action):
		new_V = np.amax(self.Q[self.s1_x][self.s1_y][:])
		self.V[self.s1_x][self.s1_y] = new_V
		old_Q = self.Q[self.s0_x][self.s0_y][action]
		#print(old_Q)
		self.Q[self.s0_x][self.s0_y][action] += eta*(reward + gamma*new_V - old_Q)
		#print(self.Q[self.s0_x][self.s0_y][action])

def main():
	e, E, T = 0, 100000, 20
	agent = Agent(11,11,4)

	while e<E:
		t = 0
		agent.reset()
		while t<T:
			a = agent.get_action()
			agent.move(a)
			reward = agent.reward()
			agent.update_arrays(reward, a)
			t+=1
		e+=1

	nx = 11
	ny = 11

	x=range(nx)
	y=range(ny)

	data = agent.V

	hf = plt.figure()
	ha = hf.add_subplot(111, projection='3d')

	X,Y = np.meshgrid(x,y)
	ha.plot_surface(X,Y, data)

	print(agent.V)

	plt.show()

if __name__ == "__main__":
	main()