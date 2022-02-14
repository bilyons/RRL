import numpy as np
import time
import random
import pickle
import matplotlib.pyplot as plt 
from PIL import Image
import cv2

class Agent:
	def __init__(self, world, epsilon, learning_rate, discount, temp, load_qtable=None):
		self.x = 0
		self.y = 0
		self.prev_finish = 0
		self.temp = 5.1
		self.width = world.width
		self.height = world.height
		self.action_space = 4
		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.discount = discount
		self.colour = (0,0,255)
		self.temp = temp
		if load_qtable is None:
			self.qtable = {}
			for x_1 in range(-self.width+1, self.width):
				for y_1 in range(-self.height+1, self.height):
					for prev in range(0, 2):
						self.qtable[(x_1, y_1), prev] = [np.random.uniform(-2,0) for i in range(self.action_space)]
		else:
			with open(f"{load_qtable}", "rb") as f:
				self.qtable = pickle.load(f)

	def __str__(self):
		return f"Blob ({self.x}, {self.y})"

	def obj_dist(self, other):
		return (self.x-other.x, self.y-other.y)

	def obj_equivalence(self, other):
		return self.x == other.x and self.y == other.y

	def reset(self):
		self.x, self.y = np.random.randint(0, self.width), np.random.randint(0, self.height)

	def observe(self):
		return (self.x, self.y), self.prev_finish

	def action_choice(self, obs):
		if np.random.rand() > self.epsilon:
			action = np.argmax(self.qtable[obs])
		else:
			action = np.random.randint(0,4)
		return action

	def boltzmann_action(self, obs):
		if np.random.rand()>self.epsilon:
			actions = np.array([0,1,2,3])
			#print(actions)
			prob = np.divide(np.exp(self.qtable[obs]), (self.temp))
			sum_prob = np.sum(prob)
			dist = np.divide(prob, sum_prob)
			action = np.random.choice(actions, p=dist)
		else:
			action = np.random.randint(0,4)
		return action

	def reward(self, goals):
		if self.x == goals[0].x and self.y == goals[0].y:
			reward = 0
			if self.prev_finish == 0:
				reward += 1
			self.prev_finish = 0
			#print(f"Goal ({self.x},{self.y})")
		elif self.x == goals[1].x and self.y == goals[1].y:
			reward = 0
			if self.prev_finish == 1:
				reward += 1
			self.prev_finish = 1
		else:
			reward = -1
		return reward

	def move(self, action):
		self.old_x = self.x
		self.old_y = self.y
		if action == 0: # Left
			if (self.x>0):
				self.x -= 1
		if action == 1: # Down
			if (self.y < (self.width-1)):
				self.y += 1
		if action == 2: # Right
			if (self.x<(self.height - 1)):
				self.x += 1
		if action == 3: # Up
			if (self.y>0):
				self.y -= 1

	def update_values(self, obs, action, reward, new_obs):
		current_q = self.qtable[obs][action]
		max_future_q = np.max(self.qtable[new_obs])
		if reward == 0:
			new_q = 0
		else:
			new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
		self.qtable[obs][action] = new_q

	def epsilon_decay(self, episode, total):
		if self.epsilon > 0.001:
			self.epsilon *= 1-(episode/total)
			#print(self.epsilon)
			if self.epsilon < 0.001:
				self.epsilon = 0.001
		else:
			self.epsilon = 0.001

class Goal:
	def __init__(self, world):
		self.width = world.width
		self.height = world.height
		self.x = 0
		self.y = 0
		self.colour = (0,255,0)

class World:
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def render(self, agents):
		img = self.get_image(agents)
		img = img.resize((300,300))
		cv2.imshow("image", np.array(img))
		cv2.waitKey(100)

	def get_image(self, agents):
		env = np.zeros((self.width, self.height, 3), dtype=np.uint8)
		env[agents[0].y][agents[0].x] = agents[0].colour # Actor
		env[agents[1].y][agents[1].x] = agents[1].colour # T1
		env[agents[2].y][agents[2].x] = agents[1].colour # T2
		img = Image.fromarray(env, 'RGB')
		return img

def main():
	EPISODES = 200_000

	# Exploration management
	SHOW_EVERY = 10_000
	world = World(5,5)
	t1 = Goal(world)
	t2 = Goal(world)
	temp = 1
	
	TEMPS=70
	for temperature in range(TEMPS):
		episode_rewards = []
		temp += 0.1
		temp = round(temp, 2)
		t1.x, t1.y, t2.x, t2.y = 0, 0, 4, 0
		agent = Agent(world, 0.9, 0.1, 0.95, temp)
		for episode in range(EPISODES):
			agent.reset()
			if episode % SHOW_EVERY == 0:
				print(f"On #{episode}, epsilon is {agent.epsilon}")
				print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
				#show = True
			else:
				show = False

			episode_reward = 0
			if show:
				world.render([agent, t1, t2])
			for i in range(15):
				obs = agent.observe()
				#act = agent.action_choice(obs)
				act= agent.boltzmann_action(obs)
				agent.move(act)
				new_obs = agent.observe()
				rew = agent.reward([t1, t2])
				agent.update_values(obs, act, rew, new_obs)
				if show:
					world.render([agent, t1, t2])

				episode_reward += rew
				
				if rew >= 0:
					#print(agent.x, agent.y)
					break


			episode_rewards.append(episode_reward)
			if episode%10_000 == 0:
				agent.epsilon_decay(episode, EPISODES)
		print(agent.qtable)
		moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode = 'valid')
		with open(f"/home/billy/Software/git/RRL/Code/trainedtemps/temp-{temp}.pickle", "wb") as f:
		    pickle.dump(agent.qtable, f)

	# plt.plot([i for i in range(len(moving_avg))], moving_avg)
	# plt.ylabel(f"Reward {SHOW_EVERY}ma")
	# plt.xlabel("episode #")
	# plt.show()


if __name__ == '__main__':
	main()
