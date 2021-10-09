from collections import defaultdict
import tensorflow as tf
import pandas as pd

from energypy import utils, memory, policy, qfunc, alpha, registry



def init_nets(env, hyp):
    actor = policy.make(env, hyp)
    onlines, targets = qfunc.make(env, size_scale=hyp['size-scale'])
    target_entropy, log_alpha = alpha.make(env, initial_value=hyp['initial-log-alpha'])
    return {
        'actor': actor,
        'online-1': onlines[0],
        'online-2': onlines[1],
        'target-1': targets[0],
        'target-2': targets[1],
        'target_entropy': float(target_entropy),
        'alpha': log_alpha,
    }


def init_writers(counters, paths):
    return {
        'random': utils.Writer('random', counters, paths['run']),
        'test': utils.Writer('test', counters, paths['run']),
        'train': utils.Writer('train', counters, paths['run']),
        'episodes': utils.Writer('episodes', counters, paths['run'])
    }


def init_optimizers(hyp):
    lr = hyp['lr']
    lr_alpha = hyp.get('lr-alpha', lr)

    return {
        'online-1': tf.keras.optimizers.Adam(learning_rate=lr),
        'online-2': tf.keras.optimizers.Adam(learning_rate=lr),
        'actor': tf.keras.optimizers.Adam(learning_rate=lr),
        'alpha': tf.keras.optimizers.Adam(learning_rate=lr_alpha),
    }


def init_fresh(hyp):
    counters = defaultdict(int)
    paths = utils.get_paths(hyp)
    transition_logger = utils.make_logger('transitions.data', paths['run'])

    metadata_path = '/content/energy_project/data/metadata.csv'
    metadata = pd.read_csv(metadata_path, index_col=0, sep=";")
    site_id = 1
    metadata = metadata[metadata.index == site_id]
    capacity = metadata['Battery_1_Capacity'][site_id] * 1000   # to W
    power_limit = metadata['Battery_1_Power'][site_id] * 1000   # to W
    charge_efficiency = metadata['Battery_1_Charge_Efficiency'][site_id]

    hyp['env']['capacity'] = float(capacity)
    hyp['env']['power'] = float(power_limit)
    hyp['env']['efficiency'] = float(charge_efficiency)

    env = registry.make(**hyp['env'], logger=transition_logger)
    print('------')
    print('env created')
    print('------')
    print('')

    buffer = memory.make(env, hyp)
    print('------')
    print('buffer created')
    print('------')
    print('')
    # Буффер создает копию файла (buffer_size x observation_space_size) забитую 0ми (np.array)
    # он пока пустой
    print(buffer.data)


    nets = init_nets(env, hyp)
    writers = init_writers(counters, paths)
    optimizers = init_optimizers(hyp)
    print('------')
    print('nets, writers, optimizers created')
    print('------')
    print('')

    target_entropy = nets.pop('target_entropy')
    hyp['target-entropy'] = target_entropy

    rewards = defaultdict(list)
    return {
        'hyp': hyp,
        'paths': paths,
        'counters': counters,
        'env': env,
        'buffer': buffer,
        'nets': nets,
        'writers': writers,
        'optimizers': optimizers,
        'transition_logger': transition_logger,
        'rewards': rewards
    }
