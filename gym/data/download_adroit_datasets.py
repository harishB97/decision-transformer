import gym
import numpy as np

import collections
import pickle

import d4rl

import os

datasets = []

"""
The *-v0 datasets were used to generate the results reported in our whitepaper and are included for backwards compatibility. However, the *-v1 datasets have improved metadata
https://github.com/Farama-Foundation/d4rl/wiki/Tasks#:~:text=The%20*%2Dv0%20datasets%20were%20used%20to%20generate%20the%20results%20reported%20in%20our%20whitepaper%20and%20are%20included%20for%20backwards%20compatibility.%20However%2C%20the%20*%2Dv1%20datasets%20have%20improved%20metadata
"""

for env_name in ['pen', 'hammer', 'door', 'relocate']:
	for dataset_type in ['expert', 'cloned', 'human']:
    # using v0 as it is the one used in the paper
		name = f'{env_name}-{dataset_type}-v0'
		env = gym.make(name)
		dataset = env.get_dataset()

		dataset_temp = {}
		dataset_temp['observations'] = dataset['observations'][:-1]
		dataset_temp['next_observations'] = dataset['observations'][1:]
		dataset_temp['actions'] = dataset['actions'][:-1]
		dataset_temp['rewards'] = dataset['rewards'][:-1]
		dataset_temp['terminals'] = dataset['terminals'][:-1]
		dataset_temp['timeouts'] = dataset['timeouts'][:-1]
		dataset_temp['timeouts'][-1] = True
		dataset = dataset_temp

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
			# 	final_timestep = (episode_step == 1000-1)
			for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
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
		print('env:',	name)
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		save_dir = '/content/gdrive/Shareddrives/DRL/decision-transformer/gym/data/d4rl-adroit'
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f'{name}.pkl')
		with open(save_path, 'wb') as f:
		  pickle.dump(paths, f)
    
		# break # downloading only expert for now
