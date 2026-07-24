# DQN CartPole 强化学习示例

这是一个完整的 DQN 强化学习示例（使用 PyTorch），在 OpenAI Gym 的 CartPole-v1 环境上训练并保存模型。

依赖
- Python 3.8+
- See requirements.txt

安装依赖
```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

运行训练
```bash
python train.py --env CartPole-v1 --episodes 500 --batch-size 64 --lr 1e-3
```

训练结束后会在 `checkpoints/` 下保存模型文件（默认名: dqn_cartpole.pth），并在控制台打印每隔若干 episode 的平均回报。

运行评估（使用已保存模型）
```bash
python eval.py --env CartPole-v1 --model checkpoints/dqn_cartpole.pth --episodes 10 --render False
```

主要文件
- train.py       : 训练入口
- eval.py        : 评估入口（加载模型进行若干回合测试）
- agent.py       : DQNAgent（行为策略、训练更新）
- model.py       : Q 网络定义
- replay_buffer.py : 经验回放池
- utils.py       : 工具函数（兼容 gym API、种子、保存）
- requirements.txt: 依赖列表

可配置项（train.py 参数）
- --env: 环境名（默认 CartPole-v1）
- --episodes: 训练 episode 数
- --batch-size: 批量大小
- --lr: 学习率
- --gamma: 折扣因子
- --epsilon-start/epsilon-final/epsilon-decay: epsilon-greedy 设置
- --target-update: 目标网络更新频率（以 step 计)
- --device: cpu / cuda

如果你希望我改成其他算法（例如 PPO、A2C、DDPG、SAC）或换环境（例如 MountainCar、LunarLander 或自定义环境），告诉我你要的算法和环境，我会把对应完整项目给你。
