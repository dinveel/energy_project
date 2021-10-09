import json
from pathlib import Path

#{'initial-log-alpha': 0.0, 'gamma': 0.99, 'rho': 0.995, 'buffer-size': 100000, 'reward-scale': 5, 'lr': 0.0003, 'batch-size': 1024, 'n-episodes': 80, 'test-every': 20, 'n-tests': 5, 'size-scale': 3, 'run-name': 'benchmark', 'env': {'name': 'pendulum', 'env_name': 'pendulum'}, 'buffer': 'new', 'target-entropy': -1.0, 'seed': 1793}
#{'run-name': 'hello', 'initial-log-alpha': 0.0, 'gamma': 0.99, 'rho': 0.995, 'buffer-size': 38400, 'reward-scale': 500, 'lr': 0.0003, 'lr-alpha': 3e-05, 'batch-size': 3840, 'n-episodes': 10, 'test-every': 256, 'n-tests': 'all', 'size-scale': 10, 'env': {'name': 'battery', 'n_batteries': 1, 'power': 75000.0, 'capacity': 300000, 'efficiency': 0.95, 'initial_charge': 0.0, 'episode_length': 96, 'dataset': {'name': 'nem-dataset'}}, 'buffer': 'new', 'target-entropy': -1.0, 'seed': 1985}

def save(data, file):
    data = data['env']
    #data = {'run-name': 'hello', 'initial-log-alpha': 0.0, 'gamma': 0.99, 'rho': 0.995, 'buffer-size': 38400, 'reward-scale': 500, 'lr': 0.0003, 'lr-alpha': 3e-05, 'batch-size': 3840, 'n-episodes': 10, 'test-every': 256, 'n-tests': 'all', 'size-scale': 10, 'env': {'name': 'battery', 'n_batteries': 1, 'power': 75000.0, 'capacity': 300000, 'efficiency': 0.95, 'initial_charge': 0.0, 'episode_length': 96, 'dataset': {'name': 'nem-dataset'}}, 'buffer': 'new', 'target-entropy': -1.0, 'seed': 2076}
    file = str(file)
    with open(file, 'w') as fi:
        json.dump(data, fi, cls=Encoder)


def load(fi):
    fi = Path.cwd() / fi
    return json.loads(fi.read_text())

class Encoder(json.JSONEncoder):
    def default(self, arg):
        if isinstance(arg, Path):
            return str(arg)
        return arg
