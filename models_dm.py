#!/usr/bin/python3

import torch
from torch import nn
from torch.nn.functional as F

class PolicyNet(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim = 256):
    super(PolicyNet, self).__init__()
    self.backbone = nn.Sequential(
      nn.Linear(state_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU()
    )
    self.mean_head = nn.Linear(hidden_dim, action_dim)
    # std must be over 0, therefore use network to predict log_std
    self.log_std_head = nn.Linear(hidden_dim, action_dim)
    self.LOG_STD_MIN = -20
    self.LOG_STD_MAX = 2
  def forward(self, state):
    hidden = self.backbone(state)
    mean = self.mean_head(hidden)
    log_std = torch.clamp(self.log_std_head(hidden), self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = torch.exp(log_std)
    return mean, std
  def sample(self, state):
    mean, std = self.forward(state)
    normal = torch.distributions.Normal(mean, std) # normal.shape = (batch, action_dim)
    x_t = normal.rsample() # x_t.shape = (batch, action_dim)
    action = torch.tanh(x_t) # action.shape = (batch, action_dim), tanh make sure that the action is in [-1, 1]
    # P(tanh(normal(mean, std))) d tanh(normal(mean, std)) = P(normal(mean, std)) d normal(mean, std)
    # P(tanh(normal(mean, std))) = P(normal(mean, std)) | det(d / dx tanh(x)) |^{-1}
    # P(tanh(normal(mean, std))) = P(normal(mean, std)) 1 / sum_i {1-tanh^2(x_i)}
    # log P(tanh(normal(mean, std))) = log P(normal(mean, std)) - log sum_i {1 - tanh^2(x_i)}
    log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6) # log_prob.shape = (batch, action_dim)
    log_prob = log_prob.sum(1, keepdim = True) # log_prob.shape = (batch, 1)
    return action, log_prob

class DiscretePolicyNet(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim = 256):
    super(DiscretePolicyNet, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(state_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, action_dim)
    )
  def forward(self, state):
    logits = self.layers(state)
    return logits
  def sample(self, state):
    logits = self.forward(state) # logits.shape = (batch, action_dim)
    probs = F.softmax(logits, dim = -1) # probs.shape = (batch, action_dim)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample() # action.shape = (batch,)
    log_prob = dist.log_prob(action).unsqueeze(dim = -1) # log_prob.shape = (batch, 1)
    return action, log_prob

class Q(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim = 256):
    super(Q, self).__init__()
    self.q1_layers = nn.Sequential(
      nn.Linear(state_dim + action_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
    self.q2_layers = nn.Sequential(
      nn.Linear(state_dim + action_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, 1)
    )
  def forward(self, state, action):
    sa = torch.cat([state, action], dim = -1) # sa.shape = (batch, state_dim + action_dim)
    q1 = self.q1_layers(sa)
    q2 = self.q2_layers(sa)
    return q1, q2

class SAC(nn.Module):
  def __init__(self, action_num, hidden_dim = 256, stack_length = 4)
