import gymnasium as gym

#Create environment

env = gym.make("LunarLander-v3")

#Sample an action

sample_action = env.action_space.sample()
print("Sample action:", sample_action)

#Sample an observation

sample_observation = env.observation_space.sample()
print("Sample observation", sample_observation)

