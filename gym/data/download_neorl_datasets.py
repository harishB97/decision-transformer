# import gym
import neorl
import numpy as np
import os

import collections
import pickle

# import d4rl


datasets = []
train_num = 1000 # num of trajectories for training, other options 100, 1000
for env_name in ['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']:
	for dataset_type in ['low', 'medium', 'high']:
		name = f'{env_name}-{dataset_type}-v3-{train_num}_trajs'
		env = neorl.make(env_name)
		dataset, _ = env.get_dataset(data_type=dataset_type, train_num=train_num) # returns train_set, val_set

		# d4rl & NeoRL uses different keywords, replacing them for consistency
		dataset['observations'] = dataset.pop('obs')
		dataset['next_observations'] = dataset.pop('next_obs')
		dataset['actions'] = dataset.pop('action')
		dataset['rewards'] = dataset.pop('reward')
		dataset['terminals'] = dataset.pop('done')

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset: # not there in NeoRL
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
        # not sure why this is required
				final_timestep = (episode_step == 1000-1) 
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
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		save_dir = '/content/gdrive/Shareddrives/DRL/decision-transformer/gym/data/offline-data/'
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f'{name}.pkl')
		with open(save_path, 'wb') as f:
			pickle.dump(paths, f)
