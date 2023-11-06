import os
os.environ['MUJOCO_GL']='egl'

import gymnasium as gym
from stable_baselines3 import PPO, AWR
import imageio
import numpy as np
from tensorboardX.utils import _prepare_video
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from collections import namedtuple

n_envs = 32
# env = gym.make("Reacher-v4", render_mode='rgb_array', width=128, height=128)
# env = DummyVecEnv([lambda: gym.make("Reacher-v4", render_mode='rgb_array', width=128, height=128)] * n)
env = SubprocVecEnv([lambda: Monitor(gym.make("Reacher-v4", render_mode='rgb_array', width=128, height=128))] * n_envs, 'fork')
callback = None
run = namedtuple('Run', ['id'])(id='dummy')
log_wandb = False

config = dict(
    total_timesteps=5000_000,
)

# # Gold standard PPO config for reacher - trains somewhat well in 1M steps
# algo = PPO
# policy_config = dict(
#     n_steps=4096 * 8 // n_envs,
#     learning_rate=3e-4,
#     n_epochs=10,
#     batch_size=64,
#     ent_coef=2e-4,
# )

# Gold standard AWR config for reacher - converges in 50k steps
algo = AWR
policy_config = dict(
    n_steps=4096 * 4 // n_envs,
    learning_rate=5e-4,
    n_epochs=10,
    batch_size=64,
    temperature=0.2,
    ent_coef=0.05,
)

if log_wandb:
    wandb_config = config.copy()
    wandb_config['algo'] = algo
    wandb_config['buffer_size'] = policy_config['n_steps'] * n_envs
    wandb_config['n_envs'] = n_envs
    wandb_config.update(policy_config)
    run = wandb.init(
        project="taskmaster_sb3",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

model = algo("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", **policy_config)
model.learn(**config, callback=callback)
if log_wandb:
    run.finish()

vec_env = model.get_env()
for i in range(n_envs):
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