# Additional MuJoCo requirements and notes

To run the MuJoCo example, install the Python bindings and viewer:

pip install mujoco mujoco-viewer imageio

Notes:
- On some systems you may need to install system libs (GL, GLFW). If pip installation fails, check MuJoCo installation docs: https://mujoco.org/
- The demo uses an offscreen renderer; on headless servers you may need to enable EGL or Xvfb.
