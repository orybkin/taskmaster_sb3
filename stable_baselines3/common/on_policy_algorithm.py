import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import math
import matplotlib.pyplot as plt
import imageio
import copy

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, MixedBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Image, Video

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        run_name: str = '',
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        relabel_ratio: float = 0,
        relabel_actor: bool = True,
        relabel_critic: bool = True,
        learning_rate_schedule: str = "",
        **kwargs
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.run_name = run_name
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = float(ent_coef)
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.relabel_ratio = relabel_ratio
        self.relabel_actor = relabel_actor
        self.relabel_critic = relabel_critic
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.learning_rate_schedule = learning_rate_schedule

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # self._setup_lr_schedule()
        self.lr_schedule = lambda *args: self.learning_rate
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    # for simplicity let's discard the true terminal obs
                    terminal_obs = self.policy.obs_to_tensor(self._last_obs)[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def change_learning_rate(self, kl_dist):
        if self.learning_rate_schedule == 'adaptive':
            min_lr = 1e-6
            max_lr = 1e-2

            if kl_dist > (2.0 * self.target_kl):
                self.learning_rate = max(self.learning_rate / 1.5, min_lr)
            if kl_dist < (0.5 * self.target_kl):
                self.learning_rate = min(self.learning_rate * 1.5, max_lr)

    def make_relabeled_buffer(self, original_buffer):
        """
        Makes a buffer where the goal is relabeled to be from the same trajectory as the original state
        """
        # TODO implement standard relabeling. Implement Geometric distribution. Check HER and LEXA. How to do with for PPO?
        relabeled_buffer = copy.deepcopy(original_buffer)

        # Get last states
        # TODO double check this works for second iteration
        obs = original_buffer.observations
        idx = original_buffer.episode_starts.astype(int)
        init_idx = idx.argmax(0)
        idx = idx.cumsum(0) * 50 - 1 + init_idx[None, :] # TODO  this is hardcoded for now
        idx = np.minimum(idx, self.n_steps - 1)
        if isinstance(obs, dict):
            last_obs = self._last_obs.copy()
            last_obs['desired_goal'] = relabeled_buffer.observations['achieved_goal'][-1]
            relabeled_buffer.observations['desired_goal'] = np.take_along_axis(obs['achieved_goal'], idx[:, :, None], 0)
            goal = relabeled_buffer.observations['desired_goal']
            pos = relabeled_buffer.observations['achieved_goal']
            goal = np.concatenate([goal[1:], last_obs['desired_goal'][None]], 0)
            pos = np.concatenate([pos[1:], last_obs['achieved_goal'][None]], 0)
            relabeled_buffer.rewards = self.env.env_method("compute_reward", pos, goal, None, indices=[0])[0]

        else:
            goals = np.take_along_axis(obs, idx[:, :, None], 0)
            # Rewards and observations. Note - these are specific to reacher
            # goal_idx = np.random.randint(original_buffer.observations.shape[1], size=original_buffer.observations.shape[0])
            # goal_obs = np.take_along_axis(original_buffer.observations, goal_idx[:, None, None], 1)
            # relabeled_buffer.observations[..., -2:] = goal_obs[:, :, -4:-2]
            relabeled_buffer.observations[..., -2:] = goals[:, :, -4:-2]
            goal = relabeled_buffer.observations[..., -2:]
            pos = relabeled_buffer.observations[..., -4:-2]

            relabeled_buffer.rewards = -np.linalg.norm(pos - goal, axis=-1) > -0.02
            last_obs = self._last_obs.copy()
            last_obs[:, -2:] = relabeled_buffer.observations[-1, :, -2:]
        self.update_values(relabeled_buffer, last_obs)

        self.relabeled_buffer = relabeled_buffer

    def update_values(self, buffer, last_obs):
        # def flatten(x): return {k: v.flatten(0, 1) for k, v in x.items()}
        # obs_tensor = obs_as_tensor(rollout_buffer.observations, self.device)
        # values = self.policy.predict_values(flatten(obs_tensor))
        def flatten(x):
            if isinstance(x, dict):
                return {k: v.flatten(0, 1) for k, v in x.items()}
            else:
                return x.flatten(0, 1)

        with th.no_grad():
            obs_tensor = obs_as_tensor(buffer.observations, self.device)
            act_tensor = obs_as_tensor(buffer.actions, self.device)
            distribution = self.policy.get_distribution(flatten(obs_tensor))
            log_probs = distribution.log_prob(act_tensor.flatten(0,1))
            buffer.log_probs = log_probs.reshape(list(act_tensor.shape[:2])).cpu().numpy()

        # Values. Is there a cleaner way to do this?
        with th.no_grad():
            values = self.policy.predict_values(flatten(obs_tensor))
            buffer.values = values.reshape(list(act_tensor.shape[:2])).cpu().numpy()
        
        done = np.concatenate([buffer.episode_starts[1:], self._last_episode_starts[None]], 0)
        buffer.rewards = buffer.rewards + self.gamma * buffer.values * done

        # Returns and advantages
        buffer.returns = np.zeros_like(buffer.returns)
        buffer.advantages = np.zeros_like(buffer.advantages)
        with th.no_grad():
            obs_tensor = obs_as_tensor(last_obs, self.device)
            values = self.policy.predict_values(obs_tensor)  
            buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            # time this
            time_start = time.time()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            time_collected = time.time()

            if not continue_training:
                break

            # import pickle
            # with open("good_ppo_data_reacher.pkl", "rb") as f: self.rollout_buffer, self._last_obs = pickle.load(f)
            # self.update_values(self.rollout_buffer, self._last_obs)

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            print(self.tensorboard_log)
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration)
                self._dump_logs()


                actor_buffer = critic_buffer = self.rollout_buffer
                if self.relabel_ratio > 0:
                    self.make_relabeled_buffer(self.rollout_buffer)
                    if self.relabel_actor or self.relabel_critic:
                        mixed_buffer = MixedBuffer(self.relabeled_buffer, self.rollout_buffer, self.relabel_ratio)
                    if self.relabel_actor:
                        actor_buffer = mixed_buffer
                    if self.relabel_critic:
                        critic_buffer = mixed_buffer
                self.actor_buffer = actor_buffer
                self.critic_buffer = critic_buffer

            self.train()
            time_trained = time.time()
            self.diagnostics['time_collect'].append(time_collected - time_start)
            self.diagnostics['time_train'].append(time_trained - time_collected)

            # Log images
            if iteration % 10 == 0 and False:
                ## Execution

                # vec_env = self.get_env()
                # obs = vec_env.reset()
                # vid = []
                # for i in range(50):
                #     action, _states = self.predict(obs, deterministic=True)
                #     obs, reward, done, info = vec_env.step(action)
                #     vid.append(vec_env.render())

                # imageio.mimwrite(f'gifs/{self.run_name}_{iteration}.gif', np.stack(vid, 0).astype(np.uint8))
                # self.logger.record("execution", Video(np.stack(vid, 0)[None].transpose([0, 1, 4, 2, 3]), 10), exclude='stdout')

                ## Value vis
                n_goal_vis = 100
                theta = np.zeros([n_goal_vis, 2])
                fingertip = np.repeat([[0.21, 0]], n_goal_vis, 0)

                # Make goal grid
                x_values = np.linspace(-0.2, 0.2, int(math.sqrt(n_goal_vis))) 
                y_values = np.linspace(-0.2, 0.2, int(math.sqrt(n_goal_vis))) 
                X, Y = np.meshgrid(x_values, y_values)
                goals = np.concatenate([X[:,:,None],Y[:,:,None]], 2).reshape([-1, 2])
                obs = np.concatenate([np.cos(theta), np.sin(theta), np.zeros_like(theta), fingertip, goals,], 1)
                obs_tensor = obs_as_tensor(obs, self.device)
                with th.no_grad(): values = self.policy.predict_values(obs_tensor).cpu().numpy()[:, 0]

                plt.figure()
                heatmap = plt.pcolormesh(X, Y, values.reshape(X.shape), shading='auto')
                plt.colorbar(heatmap)
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.savefig(f'runs/{self.run_name}/values_{iteration}.png', dpi=300)

                plt.gcf().canvas.draw()
                width, height = plt.gcf().canvas.get_width_height()
                image_as_np_array = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
                self.logger.record("values", Image(image_as_np_array, 'HWC'), exclude='stdout')

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
