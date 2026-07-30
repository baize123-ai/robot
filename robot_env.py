import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import time

class SimpleArmEnv(gym.Env):
    """
    Minimal 6-DOF arm goal-reaching env using PyBullet.
    Action: joint position increments (n,)
    Obs: joint positions, joint velocities, ee pos (3), target pos (3)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, urdf_path="kuka_iiwa/model.urdf", render=False):
        super().__init__()
        self.render_mode = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        # load an example kuka included in pybullet_data
        self.robot = p.loadURDF(urdf_path, [0,0,0], useFixedBase=True)
        # choose first 6 non-fixed joints as controllable (adjust if different model)
        self.joint_indices = [i for i in range(p.getNumJoints(self.robot))
                              if p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED][:6]
        self.n = len(self.joint_indices)
        # action: joint position delta per step
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n,), dtype=np.float32)
        # observation: q (n), qdot (n), ee_pos (3), target_pos (3)
        obs_dim = self.n + self.n + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.target = np.array([0.5, 0.0, 0.5])
        self.max_steps = 200
        self.step_counter = 0

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # reset joint states
        for j in self.joint_indices:
            p.resetJointState(self.robot, j, targetValue=0.0)
        # randomize target slightly around a nominal point
        self.target = np.array([0.5, 0.0, 0.5]) + 0.05 * np.random.randn(3)
        self.step_counter = 0
        return self._get_obs()

    def _get_ee_pos(self):
        # last controllable link as end effector
        link_id = self.joint_indices[-1]
        link_state = p.getLinkState(self.robot, link_id)
        pos = np.array(link_state[0])
        return pos

    def _get_obs(self):
        q = []
        qdot = []
        for j in self.joint_indices:
            s = p.getJointState(self.robot, j)
            q.append(s[0]); qdot.append(s[1])
        ee = self._get_ee_pos()
        obs = np.concatenate([np.array(q), np.array(qdot), ee, self.target])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # apply as position target increments with POSITION_CONTROL
        for a, j in zip(action, self.joint_indices):
            st = p.getJointState(self.robot, j)
            newpos = st[0] + float(a)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=newpos, force=200)
        # step physics (substeps to be stable)
        for _ in range(8):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(1./240.)
        obs = self._get_obs()
        ee = obs[-6:-3]
        dist = np.linalg.norm(ee - self.target)
        # reward shaping (negative distance + small control penalty)
        r_goal = -dist
        r_ctrl = -0.01 * (action**2).sum()
        done = False
        if dist < 0.05:
            r_goal += 5.0
            done = True
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True
        reward = float(r_goal + r_ctrl)
        info = {"dist": float(dist)}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass
