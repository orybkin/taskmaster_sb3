import os
os.environ['MUJOCO_GL']='egl'

import gymnasium as gym
from stable_baselines3 import PPO, AWR, PAWR
import imageio
import numpy as np
from tensorboardX.utils import _prepare_video
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from collections import namedtuple
import argparse
from datetime import datetime

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n_envs', type=int, default=32)
parser.add_argument('--relabel_actor', type=int, default=1)
parser.add_argument('--relabel_critic', type=int, default=1)
parser.add_argument('--relabel_ratio', type=float, default=.0)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--total_timesteps', type=int, default=5_000_000)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--ent_coef', type=float, default=2e-4)
parser.add_argument('--awr_coef', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--target_kl', type=float, default=None)
parser.add_argument('--max_grad_norm', type=float, default=0.5)
parser.add_argument('--n_steps', type=int, default=32768)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--algo', type=str, default='PPO')
parser.add_argument('--env_name', type=str, default='Reacher-v4')
parser.add_argument('--learning_rate_schedule', type=str, default='')
args = parser.parse_args()

if args.env_name == 'Reacher-v4':
    env_kwargs = dict(render_mode='rgb_array', width=128, height=128)
else:
    env_kwargs = dict()

n_envs = args.n_envs
if not args.debug:
    env = SubprocVecEnv([lambda: Monitor(gym.make(args.env_name, **env_kwargs))] * n_envs, 'fork')
else:
    env = DummyVecEnv([lambda: Monitor(gym.make(args.env_name, **env_kwargs))] * n_envs)
callback = None
run_name = 'dummy'
log_wandb = not args.debug

algo = dict(PPO=PPO, AWR=AWR, PAWR=PAWR)[args.algo]
config = dict(
    total_timesteps=args.total_timesteps,
)
policy_config = dict(
    n_steps=args.n_steps // n_envs,
    learning_rate=args.learning_rate,
    n_epochs=10,
    batch_size=args.batch_size,
    temperature=args.temperature,
    ent_coef=args.ent_coef,
    relabel_ratio=args.relabel_ratio,
    relabel_actor=args.relabel_actor,
    relabel_critic=args.relabel_critic,
    target_kl=args.target_kl,
    learning_rate_schedule=args.learning_rate_schedule,
    max_grad_norm=args.max_grad_norm,
    awr_coef=args.awr_coef,
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
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-") + run.name
    callback = WandbCallback(model_save_path=f"models/{run_name}",verbose=2,)

model = algo("MlpPolicy", env, verbose=1, run_name=run_name, tensorboard_log=f"runs/{run_name}", **policy_config)
model.learn(**config, callback=callback)
if log_wandb:
    run.finish()

# import pickle
# with open('good_ppo_data_reacher.pkl', 'wb') as file: pickle.dump(model.rollout_buffer, file)

## Gif
vec_env = model.get_env()
for i in range(n_envs):
    obs = vec_env.reset()
    vid = []
    for i in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vid.append(vec_env.render())

imageio.mimwrite(f'gifs/{run_name}.gif', np.stack(vid, 0).astype(np.uint8))
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

vec_env.close()