#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
  def __init__(self, action_dim, hidden_dim = 256, stack_length = 4):
    super(PolicyNet, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim)
    )
    self.mean_head = nn.Linear(hidden_dim, action_dim)
    # std must be over 0, therefore use network to predict log_std
    self.log_std_head = nn.Linear(hidden_dim, action_dim)
    self.LOG_STD_MIN = -20
    self.LOG_STD_MAX = 2
  def forward(self, states):
    encoding = self.encoding(states) # hidden.shape = (batch, hidden_dim)
    mean = self.mean_head(encoding) # mean.shape = (batch, action_dim)
    log_std = torch.clamp(self.log_std_head(encoding), self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = torch.exp(log_std) # std.shape = (batch, action_dim)
    return mean, std
  def sample(self, states):
    mean, std = self.forward(states)
    normal = torch.distributions.Normal(mean, std) # normal.shape = (batch, action_dim)
    x_t = normal.rsample() # x_t.shape = (batch, action_dim)
    actions = torch.tanh(x_t) # action.shape = (batch, action_dim), tanh make sure that the action is in [-1, 1]
    return actions
  def logprobs(self, states, actions):
    mean, std = self.forward(states)
    normal = torch.distributions.Normal(mean, std) # normal.shape = (batch, action_dim)
    x_t = torch.atanh(actions)
    # P(tanh(normal(mean, std))) d tanh(normal(mean, std)) = P(normal(mean, std)) d normal(mean, std)
    # P(tanh(normal(mean, std))) = P(normal(mean, std)) | det(d / dx tanh(x)) |^{-1}
    # P(tanh(normal(mean, std))) = P(normal(mean, std)) 1 / sum_i {1-tanh^2(x_i)}
    # log P(tanh(normal(mean, std))) = log P(normal(mean, std)) - log sum_i {1 - tanh^2(x_i)}
    logprobs = normal.log_prob(x_t) - torch.log(1 - actions.pow(2) + 1e-6) # log_prob.shape = (batch, action_dim)
    logprobs = logprobs.sum(1, keepdim = True) # log_prob.shape = (batch, 1)
    return logprobs

class DiscretePolicyNet(nn.Module):
  def __init__(self, action_num, hidden_dim = 256, stack_length = 4):
    super(DiscretePolicyNet, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim)
    )
    self.pred_head = nn.Sequential(
      nn.Linear(hidden_dim, action_num),
      nn.Softmax(dim = -1)
    )
  def forward(self, states):
    encoding = self.encoding(states)
    return self.pred_head(encoding)
  def sample(self, states):
    probs = self.forward(states) # probs.shape = (batch, action_num)
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample() # action.shape = (batch)
    return action
  def logprobs(self, states, actions):
    probs = self.forward(states) # probs.shape = (batch, action_num)
    dist = torch.distributions.Categorical(probs)
    logprobs = dist.log_prob(actions).unsqueeze(dim = -1) # logprob.shape = (batch, 1)
    return logprobs

class Q(nn.Module):
  def __init__(self, action_dim, hidden_dim = 256, stack_length = 4):
    super(Q, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim)
    )
    self.q1_head = nn.Sequential(
      nn.Linear(hidden_dim + action_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
    self.q2_head = nn.Sequential(
      nn.Linear(hidden_dim + action_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, state, action):
    encoding = self.encoding(state)
    sa = torch.cat([encoding, action], dim = -1) # sa.shape = (batch, hidden_dim + action_dim)
    q1 = self.q1_head(sa) # q1.shape = (batch, 1)
    q2 = self.q2_head(sa) # q2.shape = (batch, 1)
    return q1, q2

class DiscreteQ(nn.Module):
  def __init__(self, action_num, hidden_dim = 256, stack_length = 4):
    super(DiscreteQ, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim)
    )
    self.q1_head = nn.Sequential(
      nn.Linear(hidden_dim + action_num, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
    self.q2_head = nn.Sequential(
      nn.Linear(hidden_dim + action_num, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
    self.action_num = action_num
  def forward(self, states, actions):
    # states.shape = (batch, stack_length, h, w) actions.shape = (batch,)
    encoding = self.encoding(states)
    action = F.one_hot(actions, self.action_num) # action.shape = (batch, action_num)
    sa = torch.cat([encoding, action], dim = -1) # sa.shape = (batch, hidden_dim + action_num)
    q1 = self.q1_head(sa) # q1.shape = (batch, 1)
    q2 = self.q2_head(sa) # q2.shape = (batch, 1)
    return q1, q2

class Value(nn.Module):
  def __init__(self, hidden_dim = 256, stack_length = 4):
    super(Value, self).__init__()
    self.encoding = nn.Sequential(
      nn.Conv2d(stack_length, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.GELU(),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size = (3,3), stride = (2,2), padding = 1),
      nn.Flatten(),
      nn.Linear(hidden_dim * 14 * 14, hidden_dim)
    )
    self.value_head = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, states):
    encoding = self.encoding(states)
    values = self.value_head(encoding)
    return values

class SAC(nn.Module):
  def __init__(self, action_dim, hidden_dim = 256, stack_length = 4):
    super(SAC, self).__init__()
    self.policy = PolicyNet(action_dim, hidden_dim, stack_length)
    self.Q = Q(action_dim, hidden_dim, stack_length)
    self.V = Value(hidden_dim, stack_length)
  def act(self, states):
    actions = self.policy.sample(states) # action.shape = (batch, action_dim) log_prob = (batch, 1)
    return actions.detach()
  def pred_values(self, states):
    return self.V(states)
  def get_values(self, states, actions, log_probs, alpha = 0.2):
    # states.shape = (batch, stack_length, h, w) actions.shape = (batch, action_dim) log_probs = (batch, 1)
    q1, q2 = self.Q(states, actions) # q1.shape = (batch, 1) q2.shape = (batch, 1)
    vs = torch.minimum(q1,q2) - alpha * log_probs
    return vs.detach()
  def pred_qs(self, states, actions):
    return self.Q(states, actions)
  def get_qs(self, new_states, rewards, dones, gamma):
    # new_states.shape = (batch, stack_length, h, w) rewards.shape = (batch) dones.shape = (batch)
    vs = rewards + gamma * torch.where(dones > 0.5, torch.zeros_like(rewards), torch.ones_like(rewards)) * self.V(new_states)
    return vs.detach()
  def logprobs(self, states, actions):
    return self.policy.logprobs(states, actions)

class DiscreteSAC(nn.Module):
  def __init__(self, action_num, hidden_dim = 256, stack_length = 4):
    super(DiscreteSAC, self).__init__()
    self.policy = DiscretePolicyNet(action_num, hidden_dim, stack_length)
    self.Q = DiscreteQ(action_num, hidden_dim, stack_length)
    self.V = Value(hidden_dim, stack_length)
  def act(self, states):
    action = self.policy.sample(states) # action.shape = (batch,) log_prob = (batch, 1)
    return action.detach()
  def pred_values(self, states):
    return self.V(states)
  def get_values(self, states, actions, log_probs, alpha = 0.1):
    # states.shape = (batch, stack_length, h, w) actions.shape = (batch,) log_probs = (batch, 1)
    q1, q2 = self.Q(states, actions) # q1.shape = (batch, 1) q2.shape = (batch, 1)
    vs = torch.minimum(q1, q2) - alpha * log_probs
    return vs.detach()
  def pred_qs(self, states, actions):
    return self.Q(states, actions)
  def get_qs(self, new_states, rewards, dones, gamma):
    # new_states.shape = (batch, stack_length, h, w) rewrads.shape = (batch,) dones.shape = (batch)
    vs = rewards + gamma * torch.where(dones > 0.5, torch.zeros_like(rewards), torch.ones_like(rewards)) * self.V(new_states)
    return vs.detach()
  def logprobs(self, states, actions):
    return self.policy.logprobs(states, actions)
