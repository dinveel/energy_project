"""

dataset = AbstractDataset()

"""


from collections import OrderedDict, defaultdict
import json

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def make_perfect_forecast(prices, horizon):
    prices = np.array(prices).reshape(-1, 1)
    forecast = np.hstack([np.roll(prices, -i) for i in range(0, horizon)])
    return forecast[:-(horizon-1), :]


def load_episodes(path):  # data is like [dataFrame_1, dataFrame_2, ...]
    #  pass in list of filepaths
    if isinstance(path, list):
        if isinstance(path[0], pd.DataFrame):
            #  list of dataframes?
            return path
        else:
            #  list of paths
            episodes = [Path(p) for p in path]
            print(f'loading {len(episodes)} from list')
            csvs = [pd.read_csv(p, index_col=0, sep = ';') for p in tqdm(episodes) if p.suffix == '.csv']
            parquets = [pd.read_parquet(p) for p in tqdm(episodes) if p.suffix == '.parquet']

            eps = csvs + parquets
            print(f'loaded {len(episodes)} from list')
            return eps

    #  pass in directory
    elif Path(path).is_dir() or isinstance(path, str):
        path = Path(path)
        episodes = [p for p in path.iterdir() if p.suffix == '.csv']
    else:
        path = Path(path)
        assert path.is_file() and path.suffix == '.csv'
        episodes = [path, ]

    print(f'loading {len(episodes)} from {path.name}')
    eps = [pd.read_csv(p, index_col=0, sep = ';') for p in tqdm(episodes)]
    print(f'loaded {len(episodes)} from {path.name}')
    return eps


def round_nearest(x, divisor):
    return x - (x % divisor)

from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def get_data(self, cursor):
        #  relies on self.dataset
        return OrderedDict({k: d[cursor] for k, d in self.dataset.items()})

    def reset(self, mode=None):
        #  can dispatch based on mode, or just reset
        #  should return first obs using get_data
        return self.get_data(0)

    def setup_test(self):
        #  called by energypy.main
        #  not optional - even if dataset doesn't have the concept of test data
        #  no test data -> setup_test should return True
        return True

    def reset_train(self):
        #  optional - depends on how reset works
        raise NotImplementedError()

    def reset_test(self, mode=None):
        #  optional - depends on how reset works
        raise NotImplementedError()


class RandomDataset(AbstractDataset):
    def __init__(self, n=10, n_features=1, n_batteries=1, logger=None):
        self.dataset = self.make_random_dataset(n, n_features, n_batteries)
        self.test_done = True  #  no notion of test data for random data
        self.n_number = n
        self.reset()

    def make_random_dataset(self, n, n_features, n_batteries):
        np.random.seed(42)
        #  (timestep, batteries, features)
        prices = np.random.uniform(0, 100, n*n_batteries).reshape(n, n_batteries, 1)
        features = np.random.uniform(0, 100, n*n_features*n_batteries).reshape(n, n_batteries, n_features)
        return {'prices': prices, 'features': features}


class NEMDataset(AbstractDataset):
    def __init__(
        self,
        n_batteries,
        train_episodes=None,
        test_episodes=None,
        price_col='price_buy_00',
        logger=None
    ):  # FINE
        self.n_batteries = n_batteries
        self.price_col = price_col

        train_episodes = load_episodes(train_episodes)
        test_episodes = load_episodes(test_episodes)
        self.episodes = {
            'train': train_episodes,
            #  our random sampling done on train episodes
            'random': train_episodes,
            'test': load_episodes(test_episodes),
        }
        #print('Cawabanga __', train_episodes)
        #print('Cawabanga __', test_episodes[0]['pv_00'])
        #  want test episodes to be a multiple of the number of batteries
        episodes_before = len(self.episodes['test'])
        lim = round_nearest(len(self.episodes['test'][:]), self.n_batteries)
        self.episodes['test'] = self.episodes['test'][:lim]
        assert len(self.episodes['test']) % self.n_batteries == 0
        episodes_after = len(self.episodes['test'])
        print(f'lost {episodes_before - episodes_after} test episodes due to even multiple')

        #  test_done is a flag used to control which dataset we sample from
        #  it's a bit hacky
        self.test_done = True
        self.reset()

    def reset(self, mode='train'):
        if mode == 'test':
            return self.reset_test()
        else:
            return self.reset_train()

    def setup_test(self):  # FINE
        #  called by energypy.main
        self.test_done = False
        self.test_episodes_idx = list(range(0, len(self.episodes['test'])))
        return self.test_done

    def reset_train(self):  # FINE
                            # [[1.csv], [2.csv]] - picks (n_batts) files at the time
                            # make dataset = {'prices':, 'features':} [from [...]]
        episodes = random.sample(self.episodes['train'], self.n_batteries)

        ds = defaultdict(list)
        for episode in episodes:
            episode = episode.copy()
            prices = episode.pop(self.price_col)
            ds['prices'].append(prices.reset_index(drop=True).values.reshape(prices.shape[0], 1, 1))
            ds['features'].append(episode.reset_index(drop=True).values.reshape(prices.shape[0], 1, -1))
        #  TODO could call this episode
        self.dataset = {
            'prices': np.concatenate(ds['prices'], axis=1),
            'features': np.concatenate(ds['features'], axis=1),
        }
        return self.get_data(0)

    def reset_test(self):  # FINE
                           # picks (n_batts) files, preparing datasets
                           # when test_episodes = [] -> len = 0 -> test_done = True
        episodes = self.test_episodes_idx[:self.n_batteries]
        self.test_episodes_idx = self.test_episodes_idx[self.n_batteries:]

        ds = defaultdict(list)
        for episode in episodes:
            episode = self.episodes['test'][episode].copy()
            prices = episode.pop(self.price_buy_col)
            ds['prices'].append(prices.reset_index(drop=True))
            ds['features'].append(episode.reset_index(drop=True))

        #  TODO could call this episode
        self.dataset = {
            'prices': pd.concat(ds['prices'], axis=1).values,
            'features': pd.concat(ds['features'], axis=1).values,
        }

        if len(self.test_episodes_idx) == 0:
            self.test_done = True

        return self.get_data(0)

if __name__ == '__main__':
    n_batteries = 1

    # эти элементы для задания random-dataset
    #n = 10
    #n_features = 2
    #dataset = RandomDataset(n, n_features, n_batteries)
    #print(dataset.dataset)
    
    train_episodes = 'data/train/'
    test_episodes= 'data/submit/'
    price_col = 'price_buy_00'

    dataset = NEMDataset(n_batteries, train_episodes, test_episodes, price_col) 
    # the dataset is from reset() -> mode=train -> train_dataset (1 of them)

    print('-------------')
    print('-------------')
    print(dataset.dataset)
    #print(dataset.get_data(0))
    print('-------------')
    #print('mine dataset')
    #print(dataset.dataset)
    #print('self.episodes train: ', dataset.episodes['train'])
    #print('self.episodes test: ', dataset.episodes['test'])
    #print('elements of prices: ', len(dataset.dataset['prices']))
    #print('elements of features: ', len(dataset.dataset['features']))
