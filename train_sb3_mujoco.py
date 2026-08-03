"""Training script using Stable-Baselines3 PPO on the MuJoCo arm env.

Usage:
  python train_sb3_mujoco.py

Requirements: mujoco, mujoco-viewer, stable-baselines3, gym
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

# Import the Mujoco env from the repository
try:
    from mujoco.mujoco_env import MujocoArmEnv
except Exception:
    # allow running when PYTHONPATH not set; assume repo root
    from mujoco.mujoco_env import MujocoArmEnv


def make_env():
    return MujocoArmEnv(model_path='mujoco/arm2d.xml', render=False)


if __name__ == '__main__':
    num_envs = 4
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    total_timesteps = 200_000
    print(f"Starting training for {total_timesteps} timesteps on MuJoCo env (num_envs={num_envs})...")
    model.learn(total_timesteps=total_timesteps)
    save_path = 'ppo_mujoco_arm'
    model.save(save_path)
    print(f"Model saved to {save_path}")

    env.close()
