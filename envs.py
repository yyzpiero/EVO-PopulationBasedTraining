import os

import gym
import numpy as np
import torch
import typing
if typing.TYPE_CHECKING:
    from stable_baselines3.common.type_aliases import GymEnv
from copy import deepcopy
import pybullet_envs  
from gym.spaces.box import Box

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, VecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from envs_core import SubprocVecEnv
from gym_minigrid.wrappers import *


def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=False):
    def _thunk():
        
        env = gym.make(env_id)
        is_minigrid = env_id[0:8] == "MiniGrid"
        # is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = NoopResetEnv(env, noop_max=30)
        #     env = MaxAndSkipEnv(env, skip=4)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

            
        if is_minigrid:
            env = OneHotPartialObsWrapper(env)
            #env = RGBImgPartialObsWrapper(env)
            env = FlatObsWrapper(env)
        if isinstance(env.action_space, gym.spaces.Box):
        
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            #env = gym.wrappers.NormalizeReward(env)
            #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
            #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        
        # if is_atari:
        #     if len(env.observation_space.shape) == 3:
        #         env = EpisodicLifeEnv(env)
        #         if "FIRE" in env.unwrapped.get_action_meanings():
        #             env = FireResetEnv(env)
        #         env = WarpFrame(env, width=84, height=84)
        #         env = ClipRewardEnv(env)
        # elif len(env.observation_space.shape) == 3:
        #     raise NotImplementedError(
        #         "CNN models work only for atari,\n"
        #         "please use a custom wrapper for a custom pixel input env.\n"
        #         "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        # obs_shape = env.observation_space.shape
        # if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #     env = TransposeImage(env, op=[2, 0, 1])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _thunk

def make_vec_envs(env_name, seed, num_processes, gamma=None, sub_proc=False,log_dir=None, 
                  device="cpu", allow_early_resets=True, num_frame_stack=None,
                  no_obs_norm=False):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        if sub_proc:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if no_obs_norm == False:
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                #pass
                envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
            else:
                #pass
                envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs

def make_eval_env(env_name, seed = 53, gamma=None, no_obs_norm=False):
    envs = [make_env(env_name, seed, 0)]
    envs = DummyVecEnv(envs)

    if no_obs_norm == False:
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
            else:
                envs = VecNormalize(envs, gamma=gamma)


    return envs

# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, list):
            action_to_submit = []
            for ac in actions:
                if isinstance(ac, torch.LongTensor) or ac.dtype == torch.int64:
                    ac = ac.squeeze(1)
                action_to_submit.append(ac.cpu().numpy())
            if isinstance(self.venv, DummyVecEnv):
                action_to_submit = [action_to_submit]
            self.venv.step_async(action_to_submit)
        else:
            if isinstance(actions, torch.LongTensor):
                # Squeeze the dimension for discrete actions
                actions = actions.squeeze(1)
                actions = actions.cpu().numpy()
            #if not isinstance(actions, np.ndarray):
                
            self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


def wrap_env(env, verbose = 0, monitor_wrapper = True):
    """ "
    Wrap environment with the appropriate wrappers if needed.
    For instance, to have a vectorized environment
    or to re-order the image channels.
    :param env:
    :param verbose:
    :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
    :return: The wrapped environment.
    """
    if not isinstance(env, VecEnv):
        if not is_wrapped(env, Monitor) and monitor_wrapper:
            if verbose >= 1:
                print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])

    # Make sure that dict-spaces are not nested (not supported)
    # check_for_nested_spaces(env.observation_space)

    # if not is_vecenv_wrapped(env, VecTransposeImage):
    #     wrap_with_vectranspose = False
    #     if isinstance(env.observation_space, gym.spaces.Dict):
    #         # If even one of the keys is a image-space in need of transpose, apply transpose
    #         # If the image spaces are not consistent (for instance one is channel first,
    #         # the other channel last), VecTransposeImage will throw an error
    #         for space in env.observation_space.spaces.values():
    #             wrap_with_vectranspose = wrap_with_vectranspose or (
    #                 is_image_space(space) and not is_image_space_channels_first(space)
    #             )
    #     else:
    #         wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
    #             env.observation_space
    #         )

    #     if wrap_with_vectranspose:
    #         if verbose >= 1:
    #             print("Wrapping the env in a VecTransposeImage.")
    #         env = VecTransposeImage(env)

    return env

def is_wrapped(env, wrapper_class):
    """
    Check if a given environment has been wrapped with a given wrapper.
    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None

def unwrap_wrapper(env, wrapper_class):
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.
    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# Define here to avoid circular import
def sync_envs_normalization(env: "GymEnv", eval_env: "GymEnv") -> None:
    """
    Sync eval env and train env when using VecNormalize
    :param env:
    :param eval_env:
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, VecNormalize):
            # Only synchronize if observation normalization exists
            if hasattr(env_tmp, "obs_rms"):
                eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv