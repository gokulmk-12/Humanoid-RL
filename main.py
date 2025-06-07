import mujoco
import mujoco_viewer
import numpy as np
import gymnasium as gym
from typing import Optional
from utils.config import *
from utils.rewards import *

class HumanoidEnv(gym.Env):
    def __init__(self, robot_name: str = "g1", task_name: str = "walking", render_mode: str = "human", width: int = 1200, height: int = 700):
        super(HumanoidEnv, self).__init__()

        self.sim_dt = 0.002
        self.ctrl_dt = 0.02

        xml_file = "g1.xml"
        self.robot_name = robot_name
        self.task_name = task_name

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.key_id = self.model.key("knees_bent").id
        self.model.opt.timestep = 0.01

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model, 
                self.data, 
                title=f"Humanoid Mujoco: {self.robot_name}, Task: {self.task_name}", 
                width=width, 
                height=height)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(86, ), dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low = np.array(self.model.jnt_range[1:].T[0][:12], dtype=np.float32),
            high = np.array(self.model.jnt_range[1:].T[1][:12], dtype=np.float32)
        )

        self._init_viewer_config()
        self._get_indices()
    
    def _get_indices(self):
        self._init_q = np.array(self.model.keyframe("knees_bent").qpos)
        self._default_pose = np.array(
            self.model.keyframe("knees_bent").qpos[7:]
        )
        self._pelvis_imu_site_id = self.model.site("imu_in_pelvis").id

        self._floor_geom_id = self.model.geom("floor").id
        self._feet_geom_id = np.array(
            [self.model.geom(name).id for name in FEET_GEOMS]
        )
        self._feet_site_id = np.array([
            self.model.site(name).id for name in FEET_SITES
        ])
        self._left_foot_geom_id = self.model.geom("left_foot").id
        self._right_foot_geom_id = self.model.geom("right_foot").id
        self._left_shin_geom_id = self.model.geom("left_shin").id
        self._right_shin_geom_id = self.model.geom("right_shin").id

        foot_linvel_sensor_adr = []
        for site in FEET_SITES:
            sensor_id = self.model.sensor(f"global_linvel_{site}").id
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr+sensor_dim))
            )
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        gait_freq = 1.25
        phase_dt = 2 * np.pi * self.ctrl_dt * gait_freq
        phase = np.array([0, np.pi])

        cmd = self.sample_command()

        self.info = {
            "step": 0,
            "command": cmd,
            "last_action": np.zeros(12),
            "feet_air_time": np.zeros(2),
            "last_contact": np.zeros(2, dtype=np.bool),
            "swing_peak": np.zeros(2),
            "phase_dt": phase_dt,
            "phase": phase
        }

        contact = np.array([
            geoms_colliding(self.data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])

        obs = self._get_obs(self.data, contact)
        return obs, self.info

    def _get_obs(self, data: mujoco.MjData, contact: np.array):
        gyro = get_gyro(self.model, data, frame="pelvis")
        noisy_gyro = (
            gyro
            + (2 * np.random.uniform(size=gyro.shape)-1)
            * noise_config["level"] * noise_config["gyro"]
        )
        
        gravity = data.site_xmat[self._pelvis_imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])
        noisy_gravity = (
            gravity
            + (2 * np.random.uniform(size=gravity.shape)-1)
            * noise_config["level"] * noise_config["gravity"]
        )

        joint_angles = data.qpos[7:]
        noisy_joint_angles = (
            joint_angles
            + (2 * np.random.uniform(size=joint_angles.shape)-1)
            * noise_config["level"] * noise_config["joint_pos"]
        )

        joint_vel = data.qvel[6:]
        noisy_joint_vel = (
            joint_vel
            + (2 * np.random.uniform(size=joint_vel.shape)-1)
            * noise_config["level"] * noise_config["joint_vel"]
        )

        linvel = get_local_linvel(self.model, data, frame="pelvis")
        noisy_linvel = (
            linvel
            + (2 * np.random.uniform(size=linvel.shape)-1)
            * noise_config["level"] * noise_config["linvel"]
        )

        cos = np.cos(self.info["phase"])
        sin = np.sin(self.info["phase"])
        phase = np.concatenate([cos, sin])

        state = np.hstack([
            noisy_linvel,
            noisy_gyro,
            noisy_gravity,
            self.info["command"],
            noisy_joint_angles - self._default_pose,
            noisy_joint_vel,
            self.info["last_action"],
            phase,
        ])

        accelerometer = get_accelerometer(self.model, data, frame="pelvis")
        global_angvel = get_global_angvel(self.model, data, frame="pelvis")
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        com_height = data.qpos[2]

        privileged_state = np.hstack([
            state,
            gyro,
            accelerometer,
            gravity,
            linvel,
            global_angvel,
            joint_angles - self._default_pose,
            joint_vel,
            com_height,
            contact,
            feet_vel,
            self.info["feet_air_time"],
        ])

        return {
            "state": state,
            "privileged_state": privileged_state
        }
    
    def step(self, info, action):
        motor_action = self._default_pose.copy()
        motor_action[0:12] += action * action_scale
        self.data.ctrl[:] = motor_action

        mujoco.mj_step(self.model, self.data)

        contact = np.array([
            geoms_colliding(self.data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
        contact_filt = contact | self.info["last_contact"]
        first_contact = (self.info["feet_air_time"] > 0.0) * contact_filt
        self.info["feet_air_time"] += self.ctrl_dt
        p_f = self.data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        self.info["swing_peak"] = np.maximum(self.info["swing_peak"], p_fz)

        obs = self._get_obs(self.data, contact)
        done = self._get_termination(self.data)

        reward = self._get_reward(self.data, action, info, done, first_contact)
        rewards = {
            k: v*reward_config_scales[k] for k,v in reward.items()
        }
        task_reward, aux_reward = sum(list(rewards.values())[:2]), sum(list(rewards.values())[2:])
        reward = task_reward * np.exp(0.02 * aux_reward)

        self.info["step"] += 1
        phase_tp1 = self.info["phase"] + self.info["phase_dt"]
        self.info["phase"] = np.fmod(phase_tp1+np.pi, 2*np.pi) - np.pi
        self.info["last_action"] = action
        self.info["command"] = np.where(self.info["step"] > 500, self.sample_command(), self.info["command"])
        self.info["step"] = np.where(done | self.info["step"] > 500, 0, self.info["step"])
        self.info["feet_air_time"] *= ~contact
        self.info["last_contact"] = contact
        self.info["swing_peak"] *= ~contact

        if self.render_mode == "human":
            self.render()

        return obs, reward, done
    
    def _get_termination(self, data: mujoco.MjData):
        fall = get_gravity(self.model, data, "torso")[-1] < 0.0
        contact_termination = geoms_colliding(
            data, self._left_foot_geom_id, self._right_foot_geom_id
        ) | geoms_colliding (data, self._left_foot_geom_id, 
        self._right_shin_geom_id) | geoms_colliding(data, self._right_foot_geom_id, self._left_shin_geom_id)
        return (fall | contact_termination | np.isnan(data.qpos).any() | np.isnan(data.qvel).any())

    def _get_reward(self, data, action, info, done, first_contact):
        reward = {
            "tracking_linvel": reward_tracking_linvel(self.info["command"], get_local_linvel(self.model, data, "pelvis"), tracking_sigma=0.25),
            "tracking_angvel": reward_tracking_angvel(self.info["command"], get_gyro(self.model, data, "pelvis"), tracking_sigma=0.25),
            "linvel_z": reward_linvel_z(get_global_linvel(self.model, data, "torso"), get_global_linvel(self.model, data, "pelvis")),
            "angvel_xy": reward_rollpitch_vel(get_global_angvel(self.model, data, "torso")),
            "orientation": reward_orientation(get_gravity(self.model, data, "torso")),
            "base_height": reward_baseheight(data.qpos[2], base_target=0.5),
            "torques": reward_torques(data.actuator_force),
            "action_smoothing": reward_action_smoothing(action, info["last_action"]),
            "feet_slip": reward_feet_slip(self.model, data),
            "feet_height": reward_feet_height(info["swing_peak"], first_contact, max_foot_height=0.15),
            "feet_air_time": reward_feet_air_time(info["feet_air_time"], first_contact),
            "alive": reward_alive(),
            "termination": reward_termination(done),
            "stand_still": reward_stand_still(info["command"], data.qpos[7:], self._default_pose),
        }
        return reward
    
    def render(self):
        if self.viewer.is_alive:
            self.viewer.render()
        else:
            raise RuntimeError("Viewer is not initialized")

    def close(self):
        if self.render_mode == "human":
            if self.viewer.is_alive:
                self.viewer.close()
        else:
            pass

    def _init_viewer_config(self):
        self.viewer.cam.distance = 1.9
        self.viewer.cam.azimuth = 136.8
        self.viewer.cam.elevation = -19.3
        self.viewer.cam.lookat = np.array([-0.03217838, -0.02589756,  0.64210913])   

    def sample_command(self):
        lin_vel_x = np.random.uniform(low=-1.0, high=1.0, size=1)
        lin_vel_y = np.random.uniform(low=-0.5, high=0.5, size=1)
        ang_vel_yaw = np.random.uniform(low=-0.5, high=0.5, size=1)     

        if np.random.rand() < 0.1:
            return np.zeros(3)
        else:
            return np.array([lin_vel_x, lin_vel_y, ang_vel_yaw]).squeeze()

if __name__ == "__main__":
    env = HumanoidEnv(robot_name="g1", task_name="walking", render_mode="human")    

    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        next_obs, reward, done = env.step(info, action)

        if done:
            obs, info = env.reset()
        
