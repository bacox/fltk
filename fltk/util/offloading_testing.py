from pathlib import Path

import numpy as np
import yaml

from strategy.algorithms.Offloading import cl_algorithm, calc_similarity_matrix, normalize_class_distribution, \
    construct_similarity_matrix


def load_yaml_file(file_path: Path):
    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f'Cannot load file due to {e}')
        return None


def load_data(path: Path):
    files = [x for x in path.iterdir() if x.suffix in ['.yaml', '.yml']]
    yaml_data = [load_yaml_file(x) for x in files]
    profiling_data = [x for x in yaml_data if x is not None]
    return profiling_data


def remove_decision_data(profiling_data: dict):
    d = None
    if 'offloading-decisions' in profiling_data:
        d = profiling_data.pop('offloading-decisions')
    return profiling_data, d

def test_cl_algorithm(profiling_data: dict):
    print(profiling_data)
    p, d = remove_decision_data(profiling_data)
    sim_matrix = construct_similarity_matrix(p)
    decisions = cl_algorithm(p, sim_matrix)
    print(decisions)


def test_sim_matrix(profiling_data: dict):

    p, d = remove_decision_data(profiling_data)
    sim_matrix = construct_similarity_matrix(p)
    for row in sim_matrix:
        print(row)
    print('Done')

if __name__ == '__main__':
    base_path = Path('../').resolve() / 'profiling_data'
    profiling_data = load_data(base_path)
    # sim_matrix = test_sim_matrix(profiling_data[0])
    test_cl_algorithm(profiling_data[0])
