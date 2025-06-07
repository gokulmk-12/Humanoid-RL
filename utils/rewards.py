import mujoco
import numpy as np
from utils.config import *

## Task-Based Rewards
def reward_tracking_linvel(track_command, local_linvel, tracking_sigma=0.25):
    lin_vel_error = np.sum(np.square(track_command[:2] - local_linvel[:2]))
    return np.exp(-lin_vel_error / tracking_sigma)

def reward_tracking_angvel(track_command, angvel, tracking_sigma=0.25):
    angvel_error = np.sum(np.square(track_command[2] - angvel[2]))
    return np.exp(-angvel_error / tracking_sigma)

## Fixed-Auxiliary Rewards
def reward_linvel_z(global_linvel_torso, global_linvel_pelvis):
    return np.square(global_linvel_pelvis[2]) + np.square(global_linvel_torso[2])

def reward_rollpitch_vel(global_angvel_torso):
    return np.sum(np.square(global_angvel_torso[:2]))

def reward_orientation(torso_zaxis):
    return np.sum(np.square(torso_zaxis - np.array([0.073, 0.0, 1.0])))

def reward_torques(torques):
    return np.sum(np.square(torques))

def reward_action_smoothing(action, last_action):
    return np.sum(np.square(action - last_action))

def reward_feet_slip(model, data):
    left_foot_vel = get_global_linvel(model, data, "left_foot")
    right_foot_vel = get_global_linvel(model, data, "right_foot")
    return np.sum(np.square(left_foot_vel[:2])) + np.sum(np.square(right_foot_vel[:2]))

def reward_stand_still(commands, qpos, default_pose):
    cmd_norm = np.linalg.norm(commands)
    cost = np.sum(np.abs(qpos - default_pose))
    cost *= cmd_norm < 0.01
    return cost

## Augmented-Auxiliary Rewards
def reward_baseheight(base_height, base_target):
    return np.square(base_height-base_target)

def reward_feet_height(swing_peak, first_contact, max_foot_height):
    error = swing_peak / max_foot_height - 1.0
    return np.sum(np.square(error) * first_contact)

def reward_feet_air_time(air_time, first_contact, thresh_min=0.2, thresh_max=0.5):
    air_time = (air_time - thresh_min) * first_contact
    air_time = np.clip(air_time, max=thresh_max-thresh_min)
    return np.sum(air_time)

## Agent-Specific Rewards
def reward_termination(done):
    return done

def reward_alive():
    return np.array(1.)