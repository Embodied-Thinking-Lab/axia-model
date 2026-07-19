import gymnasium as gym
import panda_gym
import torch


def main():
    env = gym.make("PandaReach-v3", render_mode="rgb_array", renderer="Tiny")
    obs, info = env.reset()
    
    # print("obs space:", env.observation_space)
    # print("action space:", env.action_space)
    # print("obs keys:", obs.keys() if isinstance(obs, dict) else "Not dict")
    
    img = env.render()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(reward)

    state = flatten_observation(obs)

    print(state.shape)

    
def flatten_observation(obs):
    # observation is a dict with numpy arrays
    np_obs_list = [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]

    # turn the numpy arrays to torch tensors
    tensor_list = [torch.from_numpy(arr) for arr in np_obs_list]
    
    state = torch.cat(tensor_list)
    
    state = torch.flatten(state, start_dim=-1)

    return state


main()