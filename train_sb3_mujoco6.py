"""
Training script for the 6-DoF MuJoCo arm environment.
Usage: python train_sb3_mujoco6.py
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from mujoco.mujoco_env6 import MujocoArm6Env
except Exception:
    from mujoco.mujoco_env6 import MujocoArm6Env


def make_env():
    return MujocoArm6Env(model_path='mujoco/arm6d.xml', render=False)

if __name__ == '__main__':
    num_envs = 4
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=200_000)
    model.save('ppo_mujoco_arm6')
    env.save('vecnormalize_mujoco_arm6.pkl')
    env.close()
