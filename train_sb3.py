#!/usr/bin/python3

from absl import flags, app
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her import HerReplayBuffer, GoalSelectionStrategy

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_integer('batch', default = 512, help = 'number of trajectories collected parallely')
  flags.DEFINE_string('save_ckpt', default = 'ckpt.zip', help = 'path to output checkpoint')
  flags.DEFINE_string('load_ckpt', default = None, help = 'path to checkpoint resumed')
  flags.DEFINE_enum('game', default = 'box', enum_values = {'box'}, help = 'game to train with')
  flags.DEFINE_integer("steps", default = 1000000, help = 'steps for training')
  flags.DEFINE_integer("save_freq", default = 10000, help = 'save frequency')
  flags.DEFINE_string('save_path', default = 'checkpoints', help = 'checkpoint path')
  flags.DEFINE_integer('stack_length', default = 4, help = 'length of the stack')

def main(unused_argv):
  env_id = {
    'box': 'ALE/Boxing-v5'
  }[FLAGS.game]
  #env = FrameStackObservation(GrayscaleObservation(gym.make(env_id)), FLAGS.stack_length)
  env = VecFrameStack(make_atari_env(env_id, n_envs = FLAGS.batch, seed = 0), n_stack = FLAGS.stack_length)
  if FLAGS.load_ckpt is None:
    model = SAC(
      policy = 'MlpPolicy',
      env = env,
      replay_buffer_class = HerReplayBuffer,
      replay_buffer_kwargs = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
      },
      verbose = 1
    )
  else:
    model = SAC.load(FLAGS.load_ckpt, env = env)
  checkpoint_callback = CheckpointCallback(
    save_freq = FLAGS.save_freq,
    save_path = FLAGS.save_path,
    name_prefix = f"ppo_{FLAGS.game}",
    save_replay_buffer = True,
    save_vecnormalize = True,
  )
  model.learn(total_timesteps = FLAGS.steps, callback = checkpoint_callback)
  model.save(FLAGS.save_ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)
