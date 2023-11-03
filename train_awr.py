import os
os.environ['MUJOCO_GL']='egl'

import gymnasium as gym
from stable_baselines3 import PPO, AWR
import imageio
import numpy as np
from tensorboardX.utils import _prepare_video
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from collections import namedtuple

n = 4
# env = gym.make("Reacher-v4", render_mode='rgb_array', width=128, height=128)
env = DummyVecEnv([lambda: gym.make("Reacher-v4", render_mode='rgb_array', width=128, height=128)] * n)
callback = None
run = namedtuple('Run', ['id'])(id='dummy')
log_wandb = False


config = dict(
    total_timesteps=100_000,
)

# Gold standard config for cartpole - converges in 30k steps
policy_config = dict(
    n_steps=128,
    learning_rate=5e-4,
    n_epochs=10,
    batch_size=64,
)

if log_wandb:
    wandb_config = config.copy()
    wandb_config.update(policy_config)
    run = wandb.init(
        project="taskmaster_sb3",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)

model = AWR("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **policy_config)
model.learn(**config,callback=callback)
if log_wandb:
    run.finish()

vec_env = model.get_env()
for i in range(n):
    obs = vec_env.reset()
    vid = []
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vid.append(vec_env.render())

imageio.mimwrite(f'gifs/{run.id}.gif', np.stack(vid, 0).astype(np.uint8))
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

vec_env.close()