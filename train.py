import argparse
import gym
import numpy as np
from agent import DQNAgent
from utils import set_seed, env_reset, env_step
from tqdm import trange
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='CartPole-v1')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--buffer-size', type=int, default=100000)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-final', type=float, default=0.01)
    p.add_argument('--epsilon-decay', type=float, default=500)
    p.add_argument('--target-update', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--save-path', type=str, default='checkpoints/dqn_cartpole.pth')
    p.add_argument('--render', action='store_true')
    return p.parse_args()

def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    return env

def train():
    args = parse_args()
    set_seed(args.seed)
    env = make_env(args.env, args.seed)
    obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=obs,
                     action_dim=n_actions,
                     device=args.device,
                     lr=args.lr,
                     gamma=args.gamma,
                     batch_size=args.batch_size,
                     buffer_size=args.buffer_size,
                     epsilon_start=args.epsilon_start,
                     epsilon_final=args.epsilon_final,
                     epsilon_decay=args.epsilon_decay,
                     target_update=args.target_update)

    total_steps = 0
    reward_history = []

    for ep in trange(args.episodes, desc="Episodes"):
        state = env_reset(env)
        done = False
        ep_reward = 0.0
        while not done:
            if args.render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, info = env_step(env, action)
            agent.push_transition(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            ep_reward += reward
            total_steps += 1
        reward_history.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"Episode {ep+1}, episode reward: {ep_reward:.2f}, avg50: {avg:.2f}, steps: {total_steps}")

    # save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    agent.save(args.save_path)
    print("Training finished. Model saved to", args.save_path)
    env.close()

if __name__ == '__main__':
    train()
