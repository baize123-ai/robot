"""
工具函数：gym API 兼容、seed、保存模型
"""
import random
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import gym
        gym.utils.seeding.seed(seed)
    except Exception:
        pass

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# gym compatibility helpers: handle gym vs gymnasium API differences
def env_reset(env):
    # gym older returns obs
    # gymnasium returns (obs, info)
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out

def env_step(env, action):
    # older gym: obs, reward, done, info
    # new gym/gymnasium: obs, reward, terminated, truncated, info
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    else:
        raise RuntimeError("Unexpected env.step() return shape")
