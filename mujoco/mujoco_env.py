import os
import numpy as np
import gym
from gym import spaces

try:
    import mujoco
    from mujoco import MjSim, MjViewer
except Exception as e:
    mujoco = None
    MjSim = None
    print("Warning: mujoco-python import failed:", e)

class MujocoArmEnv(gym.Env):
    """
    Simple 2-DOF arm MuJoCo env. Action: joint position increments (2,)
    Obs: qpos (2), qvel (2), target_q (2)
    The model XML is expected at mujoco/arm2d.xml in repo root.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, model_path=None, render=False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required for MujocoArmEnv. Please install mujoco (pip install mujoco mujoco-viewer).")
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'mujoco', 'arm2d.xml')
        # load model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.sim = MjSim(self.model)
        self.viewer = None
        self.render_mode = render

        # joints 2
        self.n = 2
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(self.n,), dtype=np.float32)
        obs_dim = self.n + self.n + self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.target = np.zeros(self.n)
        self.max_steps = 200
        self.step_counter = 0

    def reset(self):
        # reset qpos to zeros
        self.sim.reset()
        self.sim.data.qpos[:] = np.zeros(self.model.nq)
        self.sim.data.qvel[:] = np.zeros(self.model.nv)
        self.target = np.array([0.5, -0.5]) + 0.3 * np.random.randn(self.n)
        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        qpos = np.array(self.sim.data.qpos[:self.n])
        qvel = np.array(self.sim.data.qvel[:self.n])
        obs = np.concatenate([qpos, qvel, self.target])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # apply as position targets via actuators: set qpos desired
        # current qpos
        qpos = np.array(self.sim.data.qpos[:self.n])
        new_qpos = qpos + action
        # set qpos directly
        self.sim.data.ctrl[:] = new_qpos.tolist() + [0]*(self.model.nu - self.n)
        # step simulation
        self.sim.step()

        obs = self._get_obs()
        qpos = obs[:self.n]
        dist = np.linalg.norm(qpos - self.target)
        r = -dist - 0.01 * (action**2).sum()
        done = False
        if dist < 0.05:
            r += 5.0
            done = True
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True
        info = {"jnt_dist": float(dist)}
        return obs, float(r), done, info

    def render(self, mode='human', width=640, height=480):
        if self.viewer is None:
            try:
                self.viewer = mujoco.MjRenderContextOffscreen(self.sim, 0)
            except Exception:
                # fallback to MjViewer
                self.viewer = MjViewer(self.sim)
                self.viewer._hide_overlay = True
        # offscreen context
        try:
            self.viewer.render(width, height)
            img = self.viewer.read_pixels(width, height, depth=False)[0]
            # read_pixels returns (width,height,3) in newer bindings
            # convert to HxWx3 and flip vertically if needed
            if img.shape[0] != height:
                img = np.flipud(img)
            return img
        except Exception:
            # older bindings
            try:
                self.viewer.render()
                return None
            except Exception as e:
                print("Render failed:", e)
                return None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.free()
            except Exception:
                pass
            self.viewer = None
