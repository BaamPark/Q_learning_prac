import gym
import numpy as np

env = gym.make("MountainCar-v0")
print(env.action_space.n)
