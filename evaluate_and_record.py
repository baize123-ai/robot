"""Evaluate a trained SB3 model on the MuJoCo arm env and record rollouts to GIF/MP4.

Usage examples:
  python evaluate_and_record.py --model ppo_mujoco_arm.zip --out demo.gif --episodes 3
  python evaluate_and_record.py --model ppo_mujoco_arm.zip --out demo.mp4 --episodes 1 --max-steps 400

If your policy was trained with a VecNormalize wrapper, provide --vecnormalize path to the saved wrapper file.
"""
import os
import argparse
import numpy as np
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from mujoco.mujoco_env import MujocoArmEnv
except Exception:
    from mujoco.mujoco_env import MujocoArmEnv


def make_env_fn(model_path=None, render=True):
    def _init():
        return MujocoArmEnv(model_path=model_path, render=render)
    return _init


def record_episode(model, env, max_steps=200, render_mode='rgb_array'):
    obs = env.reset()
    frames = []
    for t in range(max_steps):
        # SB3 expects batch obs for vectorized envs; for single env deterministic prediction is fine
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        # try to render
        try:
            img = env.render(mode=render_mode, width=640, height=480)
        except Exception:
            img = None
        if img is not None:
            frames.append(img)
        if done:
            break
    return frames


def save_frames(frames, out_path, fps=25):
    if len(frames) == 0:
        print('No frames to save')
        return
    ext = os.path.splitext(out_path)[1].lower()
    if ext in ['.gif', '.gif\n']:
        imageio.mimsave(out_path, frames, fps=fps)
    else:
        # save mp4 using ffmpeg writer
        writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
        for f in frames:
            writer.append_data(f)
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained SB3 model (.zip or folder)')
    parser.add_argument('--vecnormalize', default=None, help='Path to saved VecNormalize wrapper (optional)')
    parser.add_argument('--out', default='mujoco_eval.gif', help='Output filename (gif or mp4)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to record')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--model-xml', default='mujoco/arm2d.xml', help='Path to MuJoCo XML model for env')
    args = parser.parse_args()

    # create a single env for recording with rendering enabled
    env = MujocoArmEnv(model_path=args.model_xml, render=True)

    # if a VecNormalize wrapper was saved, load it
    if args.vecnormalize is not None and os.path.exists(args.vecnormalize):
        try:
            venv = VecNormalize.load(args.vecnormalize, env)
            # put in evaluation (deterministic) mode
            venv.training = False
            venv.norm_reward = False
            env = venv
            print(f'Loaded VecNormalize from {args.vecnormalize}')
        except Exception as e:
            print('Failed to load VecNormalize wrapper:', e)
            print('Continuing with raw env')

    # load model
    model = PPO.load(args.model, env=env)
    print(f'Loaded model from {args.model}')

    all_frames = []
    for ep in range(args.episodes):
        print(f'Running episode {ep+1}/{args.episodes}...')
        frames = record_episode(model, env, max_steps=args.max_steps)
        if len(frames) == 0:
            print('Warning: no frames recorded for this episode')
        all_frames.extend(frames)

    if len(all_frames) == 0:
        print('No frames captured. Check rendering support (EGL/Xvfb) on your system.')
    else:
        save_frames(all_frames, args.out)
        print(f'Saved recording to {args.out}')

    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
