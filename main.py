import gymnasium as gym
import panda_gym

env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
obs, info = env.reset()

print("obs space:", env.observation_space)
print("action space:", env.action_space)
print("obs keys:", obs.keys() if isinstance(obs, dict) else "Not dict")

img = env.render()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(reward)