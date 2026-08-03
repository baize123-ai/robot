更新：增加 6-DoF MuJoCo 模型与环境

新增文件：
- mujoco/arm6d.xml        : 简单 6-DoF serial arm MuJoCo XML（带 camera）
- mujoco/mujoco_env6.py   : 对应的 Gym 风格环境（MujocoArm6Env），自动读取 actuators/joint 数量
- train_sb3_mujoco6.py    : 使用 Stable-Baselines3 PPO 在 6-DoF 环境上训练的示例脚本

运行提示：
- 训练： python train_sb3_mujoco6.py
- 评估与录制（使用已保存模型）： python evaluate_and_record.py --model ppo_mujoco_arm6.zip --model-xml mujoco/arm6d.xml --out arm6_demo.gif
- 请确保你的系统或 Docker 镜像支持 MuJoCo 渲染（见 README_MUJOCO.md），或使用 xvfb-run 在 headless 模式下运行。
