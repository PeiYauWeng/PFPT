from Algo.fedavg import fedavg
from Algo.fedprox import fedprox
from Algo.fedopt import fedopt
from Algo.scaffold import scaffold

ALGORITHMS = [
    'fedavg',
    'fedprox',
    'fedopt',
    'scaffold'
]

def get_algorithm(algo_name):
    # return the algorithm class via a given name
    if algo_name not in globals():
        raise NotImplementedError("Given Algorithm name not found: {}".format(algo_name))
    return globals()[algo_name]