import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robot_env import SimpleArmEnv

def make_env():
    return SimpleArmEnv(urdf_path="kuka_iiwa/model.urdf", render=False)

if __name__ == "__main__":
    num_envs = 8
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99)
    model.learn(total_timesteps=200_000)  # 200k for quick test; increase for better policy
    model.save("ppo_kuka_arm")
    env.close()
