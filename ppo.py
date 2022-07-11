from tabnanny import verbose
from model import Agent, RolloutStorage
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Union, Tuple, List
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, VecEnv, VecTransposeImage
from helper import recursive_getattr
from collections import deque
from envs import make_eval_env, make_vec_envs, wrap_env, get_vec_normalize, sync_envs_normalization, VecPyTorch, VecNormalize
import time
import gym
import warnings

import copy

class PPO():
    def __init__(self, 
                 envs, 
                 num_envs=4, 
                 seed=45821,
                 #actor_critic, 
                 num_steps=2048, 
                 hidden_size=64, 
                 update_epochs=10,
                 num_minibatches=64,
                 norm_adv=True,
                 clip_coef=0.2,
                 ent_coef=0.0,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 learning_rate=3e-4, 
                 anneal_lr=False,
                 target_kl=None,  
                 #recompute_returns=False, 
                 tb_writer=None,
                 use_gae=True,
                 clip_vloss=True,
                 create_eval_env=True, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 verbose=0,
                 device='auto'):

        # Constants
        

        self.anneal_lr = anneal_lr
        self.learning_rate = learning_rate
        #self.minibatch_size = 
        self.epochs = update_epochs
        self.num_steps = num_steps
        self.deivce = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.num_minibatches = num_minibatches
        # Models
        
        self.num_envs = num_envs 
        
        if isinstance(envs, str):
            self.envs = make_vec_envs(env_name=envs, seed=seed, num_processes=self.num_envs)
        else:
            self.envs = envs
        

        
        if isinstance(self.envs, VecEnv):
            if get_vec_normalize(self.envs):
                self.obs_rms = get_vec_normalize(self.envs).obs_rms
            else:
                self.obs_rms = None
        else:
            self.obs_rms = None

        if create_eval_env:
            if isinstance(envs, str):
                self.eval_env = copy.deepcopy(make_eval_env(env_name=envs, seed=seed).unwrapped.envs[0])
            else:
                self.eval_env = copy.deepcopy(self.envs.unwrapped.envs[0])
        else:
            self.eval_env = None
        
        if verbose > 0:
            self.verbose = True
        else:
            self.verbose = False
        
        

        if isinstance(self.envs.venv, gym.vector.SyncVectorEnv):
            self.observation_space_shape = self.envs.venv.single_observation_space.shape
            self.action_space_shape = self.envs.venv.single_action_space.shape
            self.continous = isinstance(self.envs.venv.single_action_space, gym.spaces.Box)
        else:
            self.observation_space_shape = self.envs.observation_space.shape
            self.action_space_shape = self.envs.action_space.shape
            self.continous = isinstance(self.envs.action_space, gym.spaces.Box)

        self.agent = Agent(self.envs, hidden_size, self.continous).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        self.rollouts = RolloutStorage(num_steps, num_envs=self.num_envs, observation_space_shape=self.observation_space_shape, 
                        action_space_shape=self.action_space_shape, device=self.device)
        self.rollouts.to_device(device=self.deivce)
        # Writer
        self.writer = tb_writer
        self.global_step = 0

    def train(self, total_timesteps):
        self.batch_size = int(self.num_envs * self.num_steps)

        _num_updates = total_timesteps // self.batch_size
        self.rollouts.obs[0].copy_(torch.Tensor(self.envs.reset()).to(device=self.deivce))
        self.start_time =  time.time()
        #next_done = torch.zeros(self.num_envs).to(device)
        episode_rewards = deque(maxlen=10)

        for update in range(1, _num_updates + 1):
            
            # 1. Annealing the learning rate is instructed
            if self.anneal_lr:
                _frac = 1.0 - (update - 1.0) / _num_updates
                lrnow = _frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # 2. Collect Rollouts
            for step in range(self.num_steps):

                self.global_step += 1 * self.num_envs
                #self.rollouts.obs[step] = next_obs
                #self.rollouts.dones[step] = next_done

                # Sample actions
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.rollouts.obs[step])
                    
                    #self.rollouts
                    # Obser reward and next obs
                    next_obs, reward, done, infos = self.envs.step(action.cpu().numpy())
                    #obs, reward, done, infos = envs.step(action)
                    
                    
                    for info in infos:
                        if 'episode' in info.keys():
                            episode_rewards.append(info['episode']['r'])
                            if self.verbose:
                                print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                            if self.writer is not None:
                                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], self.global_step)
                                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], self.global_step)
                            break
                    # Add to rollouts
                    self.rollouts.add_traj(next_obs, reward, done, value, action, logprob)
                    self.rollouts.to_device(device=self.deivce)

            with torch.no_grad():
                # bootstrap value if not done
                next_value = self.agent.get_value(torch.Tensor(next_obs).to(self.deivce)).reshape(1, -1)
                self.rollouts.return_calculation(next_value)
            
            # Update actor critic parameters from batches
            self.minibatch_size = int(self.batch_size // self.num_minibatches)
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for e in range(self.epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs, mb_actions, mb_logprobs, mb_values, mb_advantages, mb_returns = self.rollouts.mini_batch_generator(mb_inds)
                    if not self.continous:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(mb_obs, mb_actions.long())
                    else:
                        _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(mb_obs, mb_actions)

                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    #mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            newvalue - mb_values,
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break
            if self.verbose:
                print("SPS:", int(self.global_step / (time.time() - self.start_time)))
    
    def _get_eval_env(self, eval_env):
        """
        Return the environment that will be used for evaluation.
        :param eval_env:)
        :return:
        """
        if eval_env is None:
            return wrap_env(self.eval_env)

        if eval_env is not None:
            eval_env = wrap_env(eval_env, self.verbose)
            assert eval_env.num_envs == 1
            return eval_env
    
    def act(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.device)
        return self.agent.act(obs)

    def eval(self, obs_rms=None, num_eval_episodes=5, eval_envs=None):
        
        eval_envs = VecPyTorch(VecNormalize(self._get_eval_env(eval_env=eval_envs)), self.deivce)
        #eval_envs = VecPyTorch(self._get_eval_env(eval_env=eval_envs), self.deivce)
        #eval_envs.seed = 3424
        sync_envs_normalization(self.envs, eval_envs)
        #eval_envs = gym.make('CartPole-v0')
        #eval_envs.seed(23222)
        assert eval_envs is not None
        eval_envs.training = False
        eval_episode_rewards = []
        eval_episode_length = []
        obs = eval_envs.reset()
        # eval_recurrent_hidden_states = torch.zeros(
        #     num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        # eval_masks = torch.zeros(num_processes, 1, device=device)

        while len(eval_episode_rewards) < num_eval_episodes:
        
            obs = eval_envs.reset()
            #print(obs)
            done = False
            episode_rewards = []
            episode_length = 0
            while not done:
                with torch.no_grad():
                    action = self.act(obs).cpu().numpy()
                #action = int(np.random.randint(low=0, high=env.action_space.n))
                obs, reward, done, infos = eval_envs.step(action)
                episode_rewards.append(reward)
                episode_length += 1
                
                if done:
                    if isinstance(infos, List):
                        for info in infos:
                            if 'episode' in info.keys():
                                eval_episode_rewards.append(info['episode']['r'])
                                eval_episode_length.append(info['episode']['l'])   
                    #for info in infos:
                    else:
                        if 'episode' in infos.keys():
                            eval_episode_rewards.append(infos['episode']['r'])
                            eval_episode_length.append(infos['episode']['l'])
                        else:
                            eval_episode_rewards.append(np.sum(episode_rewards))
                            eval_episode_length.append(episode_length)    

        #eval_envs.close()
        if self.verbose:
            print(" Evaluation using {} episodes: mean reward {:.5f} mean length {:.5f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards), np.mean(eval_episode_length)))
        return np.mean(eval_episode_rewards), np.mean(eval_episode_length)


    def get_parameters(self):
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).
        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return  params


    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: str = "auto"
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).
        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, torch.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.
        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.
        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["agent"]

        return state_dicts, []