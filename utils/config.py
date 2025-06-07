import mujoco
import numpy as np

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

FEET_SITES = FEET_GEOMS = ["left_foot", "right_foot"]

noise_config = {
    "level": 1.0,
    "joint_pos": 0.03,
    "joint_vel": 1.5,
    "gravity": 0.05,
    "linvel": 0.1,
    "gyro": 0.2
}

reward_config_scales = {
    "tracking_linvel": 1.0,
    "tracking_angvel": 0.75,
    "linvel_z": 0.5,
    "angvel_xy": -0.15,
    "orientation": -2.0,
    "base_height": 5.0,
    "torques": -2e-5,
    "action_smoothing": -2e-3,
    "feet_slip": -0.25,
    "feet_height": 0.1,
    "feet_air_time": 2.0,
    "alive": 1.0,
    "termination": -100.0,
    "stand_still": -1.0,
}

action_scale = 0.2

def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str):
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr: sensor_adr+sensor_dim]

def get_gravity(model: mujoco.MjModel, data: mujoco.MjData, frame: str):
    return get_sensor_data(model, data, f"{GRAVITY_SENSOR}_{frame}")

def get_global_linvel(model: mujoco.MjModel, data: mujoco.MjData, frame:str):
    return get_sensor_data(model, data, f"{GLOBAL_LINVEL_SENSOR}_{frame}")

def get_global_angvel(model: mujoco.MjModel, data: mujoco.MjData, frame:str):
    return get_sensor_data(model, data, f"{GLOBAL_ANGVEL_SENSOR}_{frame}")

def get_local_linvel(model: mujoco.MjModel, data: mujoco.MjData, frame:str):
    return get_sensor_data(model, data, f"{LOCAL_LINVEL_SENSOR}_{frame}")

def get_accelerometer(model: mujoco.MjModel, data: mujoco.MjData, frame:str):
    return get_sensor_data(model, data, f"{ACCELEROMETER_SENSOR}_{frame}")

def get_gyro(model: mujoco.MjModel, data: mujoco.MjData, frame:str):
    return get_sensor_data(model, data, f"{GYRO_SENSOR}_{frame}")

def get_collision_info(contact, geom1, geom2):
    if contact.geom.shape[0] == 0:
        return 1e4, np.zeros(3)
    
    geom_pair = np.array([geom1, geom2])
    mask = (geom_pair == contact.geom).all(axis=1)
    mask |= (geom_pair[::-1] == contact.geom).all(axis=1)

    if not mask.any():
        return 1e4, np.zeros(3)
    
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0:3]
    return dist, normal

def geoms_colliding(data, geom1, geom2):
    return get_collision_info(data.contact, geom1, geom2)[0] < 0