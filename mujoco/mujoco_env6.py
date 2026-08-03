import os
import numpy as np
import gym
from gym import spaces

try:
    import mujoco
    from mujoco import MjSim
    # prefer offscreen renderer if available
    try:
        from mujoco import MjRenderContextOffscreen
    except Exception:
        MjRenderContextOffscreen = None
    try:
        from mujoco.viewer import MjViewer
    except Exception:
        MjViewer = None
except Exception as e:
    mujoco = None
    MjSim = None
    MjRenderContextOffscreen = None
    MjViewer = None
    print("Warning: mujoco import failed:", e)

class MujocoArm6Env(gym.Env):
    """
    6-DoF serial arm MuJoCo env. Action: joint position increments (n,)
    Obs: qpos (n), qvel (n), target_q (n)
    The model XML is expected at path provided (e.g., mujoco/arm6d.xml).
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, model_path=None, render=False):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco is required for MujocoArm6Env. Please install mujoco (pip install mujoco mujoco-viewer).")
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'arm6d.xml')
        # load model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.sim = MjSim(self.model)
        self.viewer = None
        self.render_mode = render

        # determine number of actuated joints
        self.n = int(self.model.nu)
        if self.n <= 0:
            # fallback to nq
            self.n = int(self.model.nq)
        # action and obs spaces
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(self.n,), dtype=np.float32)
        obs_dim = self.n + self.n + self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # joint-space target (randomized)
        self.target = np.zeros(self.n)
        self.max_steps = 200
        self.step_counter = 0

    def reset(self):
        self.sim.reset()
        # zero positions and velocities
        self.sim.data.qpos[:] = np.zeros(self.model.nq)
        self.sim.data.qvel[:] = np.zeros(self.model.nv)
        # random joint target within small range
        self.target = 0.5 * np.random.uniform(low=-1.0, high=1.0, size=(self.n,))
        self.step_counter = 0
        return self._get_obs()

    def _get_obs(self):
        qpos = np.array(self.sim.data.qpos[:self.n])
        qvel = np.array(self.sim.data.qvel[:self.n])
        obs = np.concatenate([qpos, qvel, self.target])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        qpos = np.array(self.sim.data.qpos[:self.n])
        # interpret action as delta on joint positions
        new_qpos = qpos + action
        # set actuator controls (position actuators expect desired joint position)
        # fill ctrl array
        ctrl = np.zeros(self.model.nu)
        ctrl[:self.n] = new_qpos
        self.sim.data.ctrl[:] = ctrl
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
        # try offscreen renderer first
        try:
            if MjRenderContextOffscreen is not None:
                if self.viewer is None:
                    self.viewer = MjRenderContextOffscreen(self.sim, 0)
                self.viewer.render(width, height)
                # newer mujoco returns (rgb, depth)
                try:
                    img = self.viewer.read_pixels(width, height, depth=False)[0]
                except Exception:
                    img = self.viewer.read_pixels(width, height, depth=False)
                # ensure HxWx3
                if img is None:
                    return None
                if img.shape[0] != height:
                    img = np.flipud(img)
                return img
            else:
                # fallback to MjViewer
                if MjViewer is None:
                    return None
                if self.viewer is None:
                    self.viewer = MjViewer(self.sim)
                    try:
                        self.viewer._hide_overlay = True
                    except Exception:
                        pass
                self.viewer.render()
                # MjViewer doesn't easily provide rgb array in older bindings
                return None
        except Exception as e:
            print('Render failed:', e)
            return None

    def close(self):
        if self.viewer is not None:
            try:
                # offscreen contexts may have free() or similar
                try:
                    self.viewer.free()
                except Exception:
                    pass
            except Exception:
                pass
            self.viewer = None
