import argparse
import gym
from utils import env_reset, env_step, set_seed
from agent import DQNAgent
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='CartPole-v1')
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--render', action='store_true')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    return env

def evaluate():
    args = parse_args()
    set_seed(args.seed)
    env = make_env(args.env, args.seed)
    obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=obs, action_dim=n_actions, device=args.device)
    map_loc = torch.device(args.device)
    agent.load(args.model, map_location=map_loc)

    returns = []
    for ep in range(args.episodes):
        state = env_reset(env)
        done = False
        ep_r = 0.0
        while not done:
            if args.render:
                env.render()
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, info = env_step(env, action)
            ep_r += reward
        returns.append(ep_r)
        print(f"[Eval] Episode {ep+1}: Return {ep_r}")
    print("Average return:", sum(returns) / len(returns))
    env.close()

if __name__ == '__main__':
    evaluate()
