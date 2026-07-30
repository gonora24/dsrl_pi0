from typing import Union
from typing import Iterable, Optional
import h5py
import jax 
import gym
import gym.spaces
import numpy as np
import pickle

import copy

from jaxrl2.data.dataset import Dataset, DatasetDict, H5Dataset
import collections
from flax.core import frozen_dict

def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        chunk_size: int = 0,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity
        self.chunk_size = chunk_size

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        if chunk_size > 0:
            rewards = np.empty((self.capacity, chunk_size), dtype=np.float32)
            terminations = np.empty((self.capacity, chunk_size), dtype=np.bool_)
        else:
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            terminations = None
        masks = np.empty((self.capacity, ), dtype=np.float32)
        discount = np.empty((self.capacity, ), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
        }
        if terminations is not None:
            self.data['terminations'] = terminations

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None # this is for streaming the online data

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size) # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = np.random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]
            
            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)
            
            next_obs_dict_curr_traj = dict()
            for k in self.data['next_observations']:
                next_obs_dict_curr_traj[k] = self.data['next_observations'][k][start:end]    
            next_observations_list.append(next_obs_dict_curr_traj)
            
            actions_list.append(self.data['actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1-self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])


        
        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
            
            
        }
        return batch
        
    def insert(self, data_dict: DatasetDict):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            if self.chunk_size > 0:
                rewards = np.empty((self.capacity, self.chunk_size), dtype=np.float32)
                terminations = np.empty((self.capacity, self.chunk_size), dtype=np.bool_)
            else:
                rewards = np.empty((self.capacity, ), dtype=np.float32)
                terminations = None
            masks = np.empty((self.capacity, ), dtype=np.float32)
            discount = np.empty((self.capacity, ), dtype=np.float32)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'discount': discount,
            }
            if terminations is not None:
                data_new['terminations'] = terminations

            for x in data_new:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2


        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:                        
                    self.data[x][self.size] = data_dict[x]
        self.size += 1
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()
        
        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size = self.size,
            _traj_counter = self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)


    def restore(self, filename):
        save_dict = np.load(filename, allow_pickle=True)[0]
        # todo test this:
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']

    def load_from_hdf5(self, filename: str, chunk_reward: bool = False,
                       query_freq: int = 10, discount: float = 0.999):
        """Load trajectories from an HDF5 file into the replay buffer.

        Args:
            filename: Path to the HDF5 file written by collect_trajectories.py.
            chunk_reward: If False, insert one transition per env step. If True,
                insert transitions with a sliding window of size query_freq so
                that actions/rewards/terminations are arrays of length Q.
            query_freq: Sliding window size (Q) used when chunk_reward=True.
            discount: Base discount factor. Applied as discount^1 for step-level
                transitions, and discount^Q for chunked transitions.
        """
        with h5py.File(filename, 'r') as f:
            for demo_key in sorted(f['data'].keys()):
                demo = f['data'][demo_key]

                pixels = demo['obs/pixels'][:]                       # (T+1, H, W, C)
                state = demo['obs/state'][:] if 'obs/state' in demo else None  # (T+1, 8)
                actions = demo['actions'][:]                          # (T, action_dim)
                rewards = demo['rewards'][:]                          # (T,)
                terminations = demo['terminations'][:]                # (T,)
                masks = demo['masks'][:]                              # (T,)
                T = len(actions)

                # HDF5 saves pixels as (T+1, H, W, C) but some envs/models use a
                # camera axis, e.g. (H, W, C, num_cameras). Match replay storage.
                if 'observations' in self.data and isinstance(self.data['observations'], dict) and 'pixels' in self.data['observations']:
                    expected_pix_shape = tuple(self.data['observations']['pixels'][0].shape)
                    if tuple(pixels.shape[1:]) != expected_pix_shape:
                        if tuple(pixels.shape[1:]) + (1,) == expected_pix_shape:
                            pixels = pixels[..., None]
                if 'observations' in self.data and isinstance(self.data['observations'], dict) and 'state' in self.data['observations']:
                    expected_state_shape = tuple(self.data['observations']['state'][0].shape)
                    if tuple(state.shape[1:]) != expected_state_shape:
                        if tuple(state.shape[1:]) + (1,) == expected_state_shape:
                            state = state[..., None]

                if not chunk_reward:
                    for t in range(T):
                        obs = {'pixels': pixels[t]}
                        if state is not None:
                            obs['state'] = state[t]
                        next_obs = {'pixels': pixels[t + 1]}
                        if state is not None:
                            next_obs['state'] = state[t + 1]

                        self.insert({
                            'observations': obs,
                            'next_observations': next_obs,
                            'actions': actions[t],
                            'next_actions': actions[t + 1] if t < T - 1 else actions[t],
                            'rewards': rewards[t],
                            'masks': masks[t],
                            'discount': discount,
                        })
                else:
                    Q = query_freq
                    chunk_discount = discount ** Q
                    for t in range(T - Q + 1):
                        obs = {'pixels': pixels[t]}
                        if state is not None:
                            obs['state'] = state[t]
                        next_obs = {'pixels': pixels[t + Q]}
                        if state is not None:
                            next_obs['state'] = state[t + Q]

                        if t + 2 * Q <= T:
                            next_actions = actions[t + Q:t + 2 * Q]
                        else:
                            next_actions = actions[t:t + Q]

                        self.insert({
                            'observations': obs,
                            'next_observations': next_obs,
                            'actions': actions[t:t + Q],
                            'next_actions': next_actions,
                            'rewards': rewards[t:t + Q],
                            'terminations': terminations[t:t + Q],
                            'masks': masks[t],
                            'discount': chunk_discount,
                        })

                self.increment_traj_counter()

class H5ReplayBuffer(H5Dataset):
    def __init__(self, path: str):
        print("making buffer from hdf5 file at ", path)
        self.h5_file = h5py.File(path, 'r')
        self.data = {
            'observations': {},
            'noise': {},
            'actions': {},
        }
        self.load_from_hdf5(path)



    def load_from_hdf5(self, filename: str):
        """Load trajectories from an HDF5 file into the replay buffer.

        Args:
            filename: Path to the HDF5 file written by collect_trajectories.py.
        """
        with h5py.File(filename, 'r') as f:
            for key in sorted(f['data'].keys()):
                if key == 'obs/pixels':
                    self.data['observations']['pixels'] = f['data']['obs/pixels'][:]
                elif key == 'obs/state':
                    self.data['observations']['state'] = f['data']['obs/state'][:]
                elif key == 'noise':
                    self.data['noise'] = f['data']['noise'][:]
                elif key == 'actions':
                    self.data['actions'] = f['data']['actions'][:]
                else:
                    raise ValueError(f"Unknown key: {key}")

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)