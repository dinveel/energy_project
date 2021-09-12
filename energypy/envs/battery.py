from collections import namedtuple
import numpy as np
import pandas as pd

from energypy import registry
from energypy.envs.base import AbstractEnv


def battery_energy_balance(initial_charge, final_charge, import_energy, export_energy, losses, pv_output):
    delta_charge = final_charge - initial_charge
    balance = import_energy - (export_energy + losses + delta_charge) + pv_output 
    np.testing.assert_almost_equal(balance, 0)


def calculate_losses(delta_charge, efficiency):
    #  delta_charge = battery delta_charge charge
    delta_charge = np.array(delta_charge)
    efficiency = np.array(efficiency)
    #  account for losses / the round trip efficiency
    #  we lose electricity when we discharge
    losses = delta_charge * (1 - efficiency)
    losses = np.array(losses)
    losses[delta_charge > 0] = 0

    if (np.isnan(losses)).any():
        losses = np.zeros_like(losses)
    return np.abs(losses)


def set_battery_config(value, n_batteries):
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return np.array(value).reshape(n_batteries, 1)
    else:
        return np.full((n_batteries, 1), value).reshape(n_batteries, 1)


class BatteryObservationSpace:
    def __init__(self, dataset, additional_features):
        shape = list(dataset.dataset['features'].shape[2:])
        shape[0] += additional_features
        self.shape = tuple(shape)


class BatteryActionSpace:
    def __init__(self, n_batteries=1):
        self.n_batteries = n_batteries
        self.shape = (1, )

        #self.low = -1
        #self.high = 1
        self.low = -self.power  # 1/4 of capacity (according to metadata)
        self.high = self.power

    def sample(self):
        return np.random.uniform(-1, 1, self.n_batteries).reshape(self.n_batteries, 1)

    def contains(self, action):
        assert (action <= 1.0).all()
        assert (action >= -1.0).all()
        return True


class Battery(AbstractEnv):
    """
    data = (n_battery, timesteps, features)
    """
    def __init__(
        self,
        n_batteries=1,
        power=0.0,
        capacity=0.0,
        efficiency=0.0,
        initial_charge=0.0,
        episode_length=96,
        dataset={'name': 'nem-dataset'},
        #dataset={'name': 'random-dataset'},
        logger=None
    ):
      
        self.power = set_battery_config(power, n_batteries)
        self.capacity = set_battery_config(capacity, n_batteries)
        self.efficiency = set_battery_config(efficiency, n_batteries)
        self.initial_charge = set_battery_config(initial_charge, n_batteries)
        self.n_batteries = n_batteries

        self.episode_length = int(episode_length)

        if isinstance(dataset, dict):
            self.dataset = registry.make(
                **dataset,
                logger=logger,
                n_batteries=n_batteries,
                #### for random-dataset
                #n=10,
                #n_features=1,
                #### for nem-dataset
                train_episodes='data/train/',
                test_episodes= 'data/submit/',
                price_col = 'price_buy_00'
            )
            print(self.dataset.dataset)
        else:
            self.dataset = dataset

        self.observation_space = BatteryObservationSpace(self.dataset, additional_features=0)
        self.action_space = BatteryActionSpace(n_batteries)

        self.elements = (
            ('observation', self.observation_space.shape, 'float32'),
            ('action', self.action_space.shape, 'float32'),
            ('reward', (1, ), 'float32'),
            ('next_observation', self.observation_space.shape, 'float32'),
            ('done', (1, ), 'bool')
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])
        print(self.Transition)

    def reset(self, mode='train'):
        self.cursor = 0
        self.charge = self.get_initial_charge()

        self.dataset.reset(mode)
        self.test_done = self.dataset.test_done
        return self.get_observation()

    def get_initial_charge(self):
        #  instance check to avoid a warning that occurs when initial_charge is an array
        if isinstance(self.initial_charge, str) and self.initial_charge == "random":
            initial = np.random.uniform(0, self.capacity[0], self.n_batteries)
        else:
            initial =  self.initial_charge
        return initial.reshape(self.n_batteries, 1)

    def get_observation(self):
        data = self.dataset.get_data(self.cursor)
        features = data['features'].reshape(self.n_batteries, -1)
        return np.concatenate([features, self.charge], axis=1)

    def setup_test(self):
        self.test_done = self.dataset.setup_test()

    def step(self, action):
        action = action.reshape(self.n_batteries, 1)    # [-power;power] already in W

        # ----------------
        current_charge = self.charge

        if action > 0:
            proposed_energy = current_charge + action * self.efficiency
        else:
            proposed_energy = current_charge + action * (1. / self.efficiency)
        
        # clipping if got out of capacity limits
        proposed_energy_clipped = np.clip(proposed_energy, 0.0, self.capacity)
        delta_charge_power = proposed_energy_clipped - current_charge

        # everything is 0.25 of an hour -> converting power from 1/4 to 1 hour (to check)
        if delta_change_energy >= 0:
            delta_charge_power = delta_charge_power / ((15. / 60.) * self.efficiency)
        else:
            delta_charge_power = delta_change_energy * self.efficiency / (15. / 60.)

        actual_delta_charge_power = np.clip(delta_charge_power, -self.power, self.power)

        if actual_delta_charge_power >= 0:
            actual_delta_charge_energy = actual_delta_charge_power * (15. / 60.) * self.efficiency
        else:
            actual_delta_charge_energy = actual_delta_charge_power * (15. / 60.) / self.efficiency

        losses = calculate_losses(actual_delta_charge_energy, self.efficiency)

        net_energy = actual_delta_charge_energy + losses

        import_energy = np.zeros_like(net_energy)
        import_energy[net_energy > 0] = net_energy[net_energy > 0]

        export_energy = np.zeros_like(net_energy)
        export_energy[net_energy < 0] = np.abs(net_energy[net_energy < 0])

        #  set charge for next timestep
        self.charge = self.charge + actual_delta_charge_energy

        #  check battery is working correctly
        battery_energy_balance(current_charge, self.charge, import_energy, export_energy, losses, pv_output)

        # ----------------


        '''   Some old code (just in case of using)

        #  expect a scaled action here
        #  -1 = discharge max, 1 = charge max
        action = np.clip(action, -1, 1)
        action = action * self.power

        #  convert from power to energy, kW -> kWh
        action = action / 2

        #  charge at the start of the interval, kWh
        initial_charge = self.charge

        #  charge at end of the interval
        #  clipped at battery capacity, kWh
        final_charge = np.clip(initial_charge + action, 0, self.capacity)

        #  accumulation in battery, kWh
        #  delta_charge can also be thought of as gross_power
        delta_charge = final_charge - initial_charge

        #  losses are applied when we discharge, kWh
        losses = calculate_losses(delta_charge, self.efficiency)

        #  net of losses, kWh
        #  add losses here because in delta_charge, export is negative
        #  to reduce export, we add a positive losses
        net_energy = delta_charge + losses

        import_energy = np.zeros_like(net_energy)
        import_energy[net_energy > 0] = net_energy[net_energy > 0]

        export_energy = np.zeros_like(net_energy)
        export_energy[net_energy < 0] = np.abs(net_energy[net_energy < 0])

        #  set charge for next timestep
        self.charge = initial_charge + delta_charge

        #  check battery is working correctly
        battery_energy_balance(initial_charge, final_charge, import_energy, export_energy, losses)

        '''

        price = self.dataset.get_data(self.cursor)['prices'].reshape(self.n_batteries,  -1)
        price = np.array(price).reshape(self.n_batteries, 1)
        reward = export_energy * price - import_energy * price

        self.cursor += 1
        done = np.array(self.cursor == (self.episode_length))

        next_obs = self.get_observation()

        info = {
            'cursor': self.cursor,
            'episode_length': self.episode_length,
            'done': done,
            'charge': self.charge
        }

        return next_obs, reward, done, info


if __name__ == '__main__':
    metadata_path = '/content/energy_project/data/metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0, sep=";")
    site_id = 1
    episode_length = 96
    metadata = metadata[metadata.index == site_id]

    capacity = metadata['Battery_1_Capacity'][site_id] * 1000   # to W
    power_limit = metadata['Battery_1_Power'][site_id] * 1000   # to W
    charge_efficiency = metadata['Battery_1_Charge_Efficiency'][site_id]

    #env = Battery()
    env = Battery(power = power_limit, capacity = capacity, efficiency = charge_efficiency, episode_length = episode_length)

    obs = env.reset()

    for _ in range(2):
        act = env.action_space.sample()
        next_obs, reward, done, info = env.step(act)
