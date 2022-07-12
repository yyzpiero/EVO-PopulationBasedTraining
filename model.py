import numpy as np
import torch
import gym
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from helper import layer_init, FixedNormal, DiagGaussian

class Agent(nn.Module):
    def __init__(self, envs, hidden_size, continous):
        super().__init__()
        self.continous = continous

        if isinstance(envs.venv, gym.vector.SyncVectorEnv):
            self.observation_shape = envs.venv.single_observation_space.shape
            if not self.continous:
                self.action_shape = envs.venv.single_action_space.n
            else:
                self.action_shape = envs.venv.single_action_space.shape
        else:
            self.observation_shape = envs.observation_space.shape
            if not self.continous:
                self.action_shape = envs.action_space.n
            else:
                self.action_shape = envs.action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.observation_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        if not self.continous:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(self.observation_shape).prod(), hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_size, self.action_shape), std=0.01),
            )
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(self.observation_shape).prod(), hidden_size)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.Tanh(),
                #layer_init(nn.Linear(hidden_size, hidden_size))
                layer_init(nn.Linear(hidden_size, np.prod(self.action_shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.ones(envs.action_space.shape) * 0.0, requires_grad=True)
            #self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(self.action_shape)))
            #num_outputs = action_space.shape[0]
            #self.dist = DiagGaussian(hidden_size, np.prod(self.action_shape))


    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if not self.continous:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        else:
            action_mean = self.actor_mean(x)
            #action_logstd = self.actor_logstd.expand_as(action_mean)
            #action_std = torch.exp(action_logstd)
            #probs = self.dist(action_mean)
            #
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.rsample()
            #return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def act(self, x):
        if not self.continous:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return action
        else:
            action_mean = self.actor_mean(x)
            #action_logstd = self.actor_logstd.expand_as(action_mean)
            #action_std = torch.exp(action_logstd)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            #action_std = torch.ones_like(action_mean) * self.actor_logstd.exp()
            probs = Normal(action_mean, action_std)
            #probs = self.dist(action_mean)
            action = probs.rsample()
            return action



class RolloutStorage:
    def __init__(self, num_steps, num_envs, observation_space_shape, action_space_shape, device):
        
        self.num_steps =num_steps
        self.step = 0

        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        #self.device = device

        self.obs = torch.zeros((self.num_steps+1, num_envs) + self.observation_space_shape)
        self.actions = torch.zeros((self.num_steps, num_envs) + self.action_space_shape)
        self.logprobs = torch.zeros((self.num_steps, num_envs))
        self.rewards = torch.zeros((self.num_steps, num_envs))
        self.dones = torch.zeros((self.num_steps+1, num_envs))
        self.values = torch.zeros((self.num_steps, num_envs))

        self.returns = torch.zeros((self.num_steps, num_envs))
        #self.advantages = torch.zeros_like(self.rewards)


    def to_device(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.actions = self.actions.to(device)
        self.logprobs = self.logprobs.to(device)
        self.returns = self.returns.to(device)
        #self.advantages = self.advantages.to(device)

    def add_traj(self, next_obs, reward, done, value, action, logprob):
        if isinstance(next_obs, np.ndarray):
            self.obs[self.step+1].copy_(torch.from_numpy(next_obs).float())
        else:
            self.obs[self.step+1].copy_(next_obs)
        
        if isinstance(reward, np.ndarray):
            self.rewards[self.step].copy_(torch.from_numpy(reward).float())
        else:
            self.rewards[self.step].copy_(reward.view(-1))
               
        self.dones[self.step+1].copy_(torch.tensor(done))
        self.values[self.step].copy_(value.flatten())
        self.actions[self.step].copy_(action)
        self.logprobs[self.step].copy_(logprob)

        self.step = (self.step + 1) % self.num_steps
        #print(self.step)

    def return_calculation(self, next_value, gamma=0.99, gae_lambda=0.95, use_gae=True):
        #next_obs = self.obs[self.step]
        #next_value = agent.get_value(next_obs).reshape(1, -1)
        
        if use_gae:
            self.advantages = torch.zeros_like(self.rewards).to(self.rewards.device)
            _lastgaelam = 0
            for t in reversed(range(self.rewards.size(0))):
                if t == self.rewards.size(0) - 1:
                    #nextnonterminal = 1.0 - self.next_done_mask
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                _delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = _lastgaelam = _delta + gamma * gae_lambda * nextnonterminal * _lastgaelam
                self.returns[t] = self.advantages[t] + self.values[t]
        else:
            #returns = torch.zeros_like(self.rewards).to(device)
            for t in reversed(range(self.rewards.size(0))):
                if t == self.rewards.size(0) - 1:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = self.returns[t + 1]
                self.returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
            self.advantages = self.returns - self.values
        
        #return returns

    def mini_batch_generator(self, mb_inds):
        b_obs = self.obs.reshape((-1,) + self.observation_space_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_space_shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        return b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], b_values[mb_inds], b_advantages[mb_inds], b_returns[mb_inds]
    