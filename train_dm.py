#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import random
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
import ale_py
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from models_dm import DiscreteSAC

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt.pth', help = 'path to checkpoint')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_string('logdir', default = 'logs', help = 'path to log directory')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')
  flags.DEFINE_integer('steps', default = 10000, help = 'number of steps per epoch')
  flags.DEFINE_integer('batch', default = 32, help = 'number of trajectories collected parallely')
  flags.DEFINE_integer('traj_length', default = 256, help = 'maximum length of a trajectory')
  flags.DEFINE_integer('epochs', default = 300, help = 'number of epoch')
  flags.DEFINE_integer('replay_buffer_size', default = 10000, help = 'replay buffer size')
  flags.DEFINE_float('gamma', default = 0.95, help = 'gamma value')
  flags.DEFINE_float('alpha', default = 0.1, help = 'alpha value')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def preprocess(img):
  # img.shape = (c = 4, H, W)
  data = np.stack([cv2.resize(frame, (224,224)) for frame in img], axis = 0) # img.shape = (4, h, w)
  return data

def main(unused_argv):
  gym.register_envs(ale_py)
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  envs = SyncVectorEnv([lambda: FrameStackObservation(GrayscaleObservation(gym.make(env_id)), FLAGS.stack_length) for _ in range(FLAGS.batch)])
  sac = DiscreteSAC(action_num = envs.single_action_space.n, stack_length = FLAGS.stack_length).to(FLAGS.device)
  criterion = nn.MSELoss().to(FLAGS.device)
  optimizer = Adam(sac.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  tb_writer = SummaryWriter(log_dir = FLAGS.logdir)
  global_steps = 0
  replay_buffer = list()
  if exists(FLAGS.ckpt):
    ckpt = torch.load(FLAGS.ckpt)
    global_steps = ckpt['global_steps']
    sac.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    replay_buffer = ckpt['replay_buffer']
  for epoch in tqdm(range(FLAGS.epochs), desc = "epoch"):
    step_pbar = tqdm(range(FLAGS.steps), desc = "step", leave = False)
    for step in step_pbar:
      obs, info = envs.reset()
      obs = np.stack([preprocess(ob) for ob in obs], axis = 0).astype(np.float32)
      # 1) sample 256 steps from a trajectory
      rollout_pbar = tqdm(range(FLAGS.traj_length), desc = "rollout", leave = False)
      for _ in rollout_pbar:
        inputs = torch.from_numpy(obs).to(next(sac.parameters()).device)
        actions = sac.act(inputs) # actions.shape = (batch,)
        actions = actions.cpu().numpy()
        new_obs, rewards, terminates, truncates, infos = envs.step(actions)
        new_obs = np.stack([preprocess(ob) for ob in new_obs], axis = 0).astype(np.float32)
        replay_buffer.append((obs, actions, new_obs, rewards.astype(np.float32), terminates))
        if len(replay_buffer) > FLAGS.replay_buffer_size: replay_buffer = replay_buffer[-FLAGS.replay_buffer_size:]
        obs = new_obs
      # 2) train with replay buffer
      trainset = random.choices(replay_buffer, k = 100)
      train_pbar = tqdm(trainset, desc = "train", leave = False)
      for o, a, no, r, d in train_pbar:
        states = torch.from_numpy(o).to(next(sac.parameters()).device)
        actions = torch.from_numpy(a).to(next(sac.parameters()).device)
        new_states = torch.from_numpy(no).to(next(sac.parameters()).device)
        rewards = torch.from_numpy(r).to(next(sac.parameters()).device)
        dones = torch.from_numpy(d).to(next(sac.parameters()).device)

        logprobs = sac.logprobs(states, actions)
        pred_q1, pred_q2 = sac.pred_qs(states, actions) # pred_q1.shape = (batch, 1) pred_q2.shape = (batch, 1)
        true_q = sac.get_qs(new_states, rewards, dones, FLAGS.gamma) # true_q.shape = (batch, 1)
        pred_v = sac.pred_values(states) # pred_v.shape = (batch, 1)
        true_v = sac.get_values(states, actions, logprobs, alpha = FLAGS.alpha) # true_v.shape = (batch, 1)

        loss = - (torch.minimum(pred_q1, pred_q2) - FLAGS.alpha * logprobs) + 0.5 * (criterion(pred_q1, true_q) + criterion(pred_q2, true_q)) + criterion(pred_v, true_v)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_pbar.set_postfix(loss = loss.detach().cpu().numpy())
        tb_writer.add_scalar('loss', loss.detach().cpu().numpy(), global_steps)
        global_steps += 1
    scheduler.step()
    ckpt = {
      'global_steps': global_steps,
      'state_dict': sac.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler
    }
    torch.save(ckpt, FLAGS.ckpt)
  envs.close()

if __name__ == "__main__":
  add_options()
  app.run(main)
