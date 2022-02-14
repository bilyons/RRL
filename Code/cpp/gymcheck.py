import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

print(len(env.observation_space.high))

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
x = [5]*2
print(x)
print(DISCRETE_OS_SIZE)
print("hello")