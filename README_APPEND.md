更新：增加 MuJoCo 仿真环境与可视化示例

新增文件
- mujoco/arm2d.xml  : 简单 2-DOF MuJoCo 模型（XML）
- mujoco/mujoco_env.py : 基于 MuJoCo 的 Gym 风格环境（MujocoArmEnv）
- mujoco/viz_mujoco.py : demo 脚本，运行 env 并把渲染帧保存为 GIF
- mujoco/README.md : MuJoCo 运行说明
- README_MUJOCO.md : 顶层 MuJoCo 安装/运行提示
- requirements.txt : 添加 mujoco, mujoco-viewer, imageio

使用说明（快速）
1) 安装额外依赖：
   pip install -r requirements.txt
   或单独： pip install mujoco mujoco-viewer imageio
2) 在仓库根目录运行 demo：
   python mujoco/viz_mujoco.py
   这会生成 `mujoco_demo.gif`（如果渲染和 offscreen 支持正常）。

注意事项
- MuJoCo 绑定在不同系统/版本下的可用性与渲染 API 存在差异。若发生渲染错误，请参阅 mujoco 官方安装说明或使用 Xvfb / EGL 作为 headless 渲染方案。
