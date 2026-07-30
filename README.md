# rl-robot-arm

Minimal end-to-end example for robot arm trajectory planning using Reinforcement Learning (PyBullet + Stable-Baselines3).

Features
- Gym-style PyBullet environment (robot_env.py)
- Training script using Stable-Baselines3 PPO (train_sb3.py)
- Dockerfile for reproducible local runs
- Colab notebook that installs deps, writes files and runs a quick training session

Quickstart — 本地（Python 环境）
1. 克隆或把本仓库下载到本地。
2. 建议创建虚拟环境并激活：
   python -m venv venv && source venv/bin/activate
3. 安装依赖：
   pip install -r requirements.txt
4. 运行训练（快速测试）：
   python train_sb3.py

Docker 快速运行（容器）
1. 构建镜像：
   docker build -t rl-robot-arm .
2. 运行容器（交互式）：
   docker run --rm -it --name rl-arm rl-robot-arm bash
3. 容器内运行训练：
   python train_sb3.py

Colab（在线一键运行）
1. 打开 Colab，新建 Notebook。
2. 将本仓库的 `run_in_colab.ipynb` 上传到 Colab（或直接复制 notebook 内容），然后按序运行 cell 即可（其中会安装依赖并写入 env 与训练脚本）。
3. 注意：Colab 环境受资源限制，训练步数已设置为快速测试级别，可根据需要调大。

说明与扩展
- 该示例使用 PyBullet 附带的 KUKA URDF（kuka_iiwa/model.urdf）。若你有自己的 URDF，请替换 robot_env.py 中的路径。
- 当前控制模式：输出为关节位置增量，底层用 POSITION_CONTROL 执行，适合初学与安全调试。
- 若要更高效训练或做视觉策略，可扩展为：SAC + HER、使用图像 observation（摄像头）、或迁移到 Isaac Gym（大并行）。
