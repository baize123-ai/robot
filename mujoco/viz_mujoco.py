"""
Run a demo in the MuJoCo env, render frames and save a GIF showing joint trajectories.
"""
import os
import numpy as np
from mujoco.mujoco_env import MujocoArmEnv

try:
    from mujoco.mujoco_env import MujocoArmEnv as TestEnv
except Exception:
    # Import from local path
    from mujoco.mujoco_env import MujocoArmEnv as TestEnv

import imageio

def run_demo(save_path='mujoco_demo.gif', steps=200):
    env = TestEnv(render=True)
    obs = env.reset()
    frames = []
    for i in range(steps):
        # simple PD-like controller: move joints toward target in obs[-2:]
        target = obs[-2:]
        qpos = obs[:2]
        action = 0.2 * (target - qpos)
        obs, r, done, info = env.step(action)
        img = env.render(mode='rgb_array', width=640, height=480)
        if img is not None:
            frames.append(img)
        if done:
            obs = env.reset()
    if len(frames) > 0:
        imageio.mimsave(save_path, frames, fps=25)
        print('Saved demo to', save_path)
    else:
        print('No frames recorded; rendering may not be supported in your mujoco installation.')
    env.close()

if __name__ == '__main__':
    run_demo()
