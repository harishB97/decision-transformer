import gym
import numpy as np

import collections
import pickle

import d4rl
import h5py

import os

datasets = []

hdf5_file = '/content/gdrive/Shareddrives/DRL/adroit-files/pen-v0_expert_clipped.hdf5'
name = os.path.splitext(os.path.basename(hdf5_file))[0]

# for env_name in ['pen', 'hammer', 'door', 'relocate']:
# 	for dataset_type in ['expert', 'cloned', 'human']:
# name = 'pen-expert-v0'
# env = gym.make(name)
# dataset = env.get_dataset()

with h5py.File(hdf5_file, 'r') as f:
  dataset = {}
  dataset['observations'] = f['observations'][:-1]
  dataset['next_observations'] = f['observations'][1:]
  dataset['actions'] = f['actions'][:-1]
  dataset['rewards'] = f['rewards'][:-1]
  dataset['terminals'] = f['terminals'][:-1]
  dataset['timeouts'] = f['timeouts'][:-1]
  dataset['timeouts'][-1] = True

  N = dataset['rewards'].shape[0]
  data_ = collections.defaultdict(list)

  use_timeouts = False
  if 'timeouts' in dataset:
    use_timeouts = True

  episode_step = 0
  paths = []
  for i in range(N):
    done_bool = bool(dataset['terminals'][i])
    if use_timeouts:
      final_timestep = dataset['timeouts'][i]
    # else: # this makes sense only for mujoco tasks where the max episode length is 1000
    #   final_timestep = (episode_step == 1000-1)
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'timeouts']:
      data_[k].append(dataset[k][i])
    if done_bool or final_timestep:
      episode_step = 0
      episode_data = {}
      for k in data_:
        episode_data[k] = np.array(data_[k])
      paths.append(episode_data)
      data_ = collections.defaultdict(list)
    episode_step += 1

  returns = np.array([np.sum(p['rewards']) for p in paths])
  num_samples = np.sum([p['rewards'].shape[0] for p in paths])
  print(f'Number of samples collected: {num_samples}')
  print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

  save_dir = '/content/gdrive/Shareddrives/DRL/decision-transformer/gym/data/d4rl-adroit'
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f'{name}.pkl')
  with open(save_path, 'wb') as f:
    pickle.dump(paths, f)
