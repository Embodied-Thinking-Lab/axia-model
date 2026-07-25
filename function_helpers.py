import torch


def compute_goal_distance(obs):
    achieved_goal = torch.from_numpy(obs["achieved_goal"])
    desired_goal = torch.from_numpy(obs["desired_goal"])
    
    goal_distance = torch.norm(achieved_goal - desired_goal)
    return goal_distance


def compute_object_height(obs):
    object_pos = torch.from_numpy(obs["achieved_goal"])
    return object_pos[2]


def compute_gripper_to_object_distance(obs):
    gripper_pos = torch.from_numpy(obs["observation"][:3])
    object_pos = torch.from_numpy(obs["achieved_goal"])
    return torch.norm(gripper_pos - object_pos)


def make_action_from_target(current_pos, target_pos, action_dim, gripper_command, xy_gain=6.0,z_gain=6.0):
    delta = target_pos - current_pos
    scaled_delta = torch.tensor(
        [
            delta[0] * xy_gain,
            delta[1] * xy_gain,
            delta[2] * z_gain,
        ],
        dtype=torch.float32,
    )

    action = torch.zeros(action_dim, dtype=torch.float32)
    action[:3] = torch.clamp(scaled_delta, -1.0, 1.0)

    if action_dim > 3:
        action[3] = gripper_command

    return action


def compute_recovery_action_target(obs, action_dim, approach_height=0.08, grasp_height=0.01, lift_height=0.12, xy_gain=6.0, 
z_gain=6.0, xy_align_tolerance=0.02, above_z_tolerance=0.02, grasp_distance_tolerance=0.025,):
                                    
    gripper_pos = torch.from_numpy(obs["observation"][:3]).to(torch.float32)
    object_pos = torch.from_numpy(obs["achieved_goal"]).to(torch.float32)

    above_object = object_pos + torch.tensor([0.0, 0.0, approach_height], dtype=torch.float32)
    at_object = object_pos + torch.tensor([0.0, 0.0, grasp_height], dtype=torch.float32)
    lift_target = gripper_pos + torch.tensor([0.0, 0.0, lift_height], dtype=torch.float32)

    xy_distance = torch.norm(gripper_pos[:2] - object_pos[:2])
    above_z_distance = torch.abs(gripper_pos[2] - above_object[2])
    gripper_to_object_distance = torch.norm(gripper_pos - object_pos)

    if xy_distance > xy_align_tolerance or above_z_distance > above_z_tolerance:
        target_pos = above_object
        gripper_command = 1.0
    elif gripper_to_object_distance > grasp_distance_tolerance:
        target_pos = at_object
        gripper_command = 1.0
    else:
        target_pos = lift_target
        gripper_command = -1.0

    return make_action_from_target( gripper_pos, target_pos, action_dim, gripper_command, xy_gain=xy_gain, z_gain=z_gain)
    

def flatten_observation(obs):
    # observation is a dict with numpy arrays
    np_obs_list = [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]

    # turn the numpy arrays to torch tensors
    tensor_list = [torch.from_numpy(arr) for arr in np_obs_list]
    state = torch.cat(tensor_list)

    return state
