import copy
import time

import yaml

from fltk.strategy.algorithms.Alg import FederatedAlgorithm
from fltk.util.config import Config
from typing import List
from scipy.stats import wasserstein_distance
import numpy as np
from itertools import zip_longest


def client_matching(client_data, performance_data):
    metrics = [list(x['pm']) for x in performance_data.values()]
    # mct = np.mean(list(performance_data.values()))
    return [(client_data[0].name, client_data[1].name)]

# Offloading util functions
def client_similarity_new(clientA: dict, clientB: dict):
    distsA = [i for i in range(len(clientA['ndd']))]
    distsB = [i for i in range(len(clientB['ndd']))]
    return wasserstein_distance(distsA, distsB, clientA['ndd'], clientB['ndd'])


def calc_similarity_matrix(clients: dict):
    m = np.zeros((len(clients), len(clients)))
    for i, client_A in enumerate(clients.items()):
        for j, client_B in enumerate(clients.items()):
            m[i][j] = client_similarity_new(client_A[1], client_B[1])
    return m


def normalize_class_distribution(class_amounts):
    norm = np.linalg.norm(class_amounts)
    return np.array(class_amounts) / norm


def sort_clients(clients: List[dict], reduced: bool = False):
    if reduced:
        return sorted(clients, key=lambda x: x['rect'], reverse=True)
    return sorted(clients, key=lambda x: x['ect'], reverse=False)


def calc_compute_times(x: dict, client_id: str):
    x['cut'] = sum(x['pm'])
    x['rcut'] = sum(x['pm'][:-1])
    x['rf'] = x['rcut'] / x['cut']
    x['ect'] = x['cut'] * (x['rl'] + x['ps'])
    x['rect'] = x['rcut'] * (x['rl'] + x['ps'])
    x['id'] = client_id
    return x


def _mean_compute_time(perf_data: dict):
    compute_times = [sum(x['pm']) * (x['rl'] + x['ps']) for x in perf_data.values()]
    return np.mean(compute_times)


def cl_algorithm(performance_data: dict, similarity_matrix: np.ndarray):
    '''

    :param performance_data: Dictionary indexed by the client_id containing the following properties:
    - profiling_data (estimates for each profiling phase (ff, fc, bc, bf))
    - Profiling size: number of local updates used for profiling
    - Batch size: local batch size of the client
    - Remaining local updates: number of remaining local updates before the round ends
    :param similarity_matrix: an NxN matrix containing pair-wise similarities of each client to each other client
    :return: A set of offloading decisions. This should contain the nodes A and B that offload from (A)
    and offloads to (B). Also should contain the time when the model needs to be offloaded.
    '''

    # Mean compute time:
    mct = _mean_compute_time(performance_data)
    for key, item in performance_data.items():
        performance_data[key] = calc_compute_times(item, key)
    slow_clients = [v for k,v in performance_data.items() if sum(v['pm']) * (v['rl'] + v['ps']) > mct]
    fast_clients = [v for k,v in performance_data.items() if sum(v['pm']) * (v['rl'] + v['ps']) <= mct]
    sorted_slow = sort_clients(slow_clients, reduced=True)
    sorted_fast = sort_clients(fast_clients, reduced=False)
    # min_matches = min(len(sorted_slow), len(sorted_fast))

    decisions = []

    # For now stick with the reverse sorted matching
    for item in zip_longest(sorted_slow, sorted_fast):
        if not None in item:
            decisions.append({
                'from': item[0]['id'],
                'to': item[1]['id'],
                'when': 0
            })
    return decisions



class Offloading(FederatedAlgorithm):
    name: str = 'offloading'

    def init_alg(self, federator_state, alg_state: dict, config: Config):
        pass

    def hook_post_startup(self, federator_state, alg_state: dict):
        pass

    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        pass

    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        pass

    def hook_training(self, federator_state, alg_state: dict, training_start_time) -> bool:
        # federator_state.logger.info('Inside training hook!')
        client_ids = [x.name for x in federator_state.selected_clients]
        if 'first_cycle' not in alg_state:
            for client_id in client_ids:
                federator_state.message_async(client_id, 'lock')
            alg_state['first_cycle'] = False

        if 'inactive' not in alg_state and all(item in federator_state.performance_data for item in client_ids):
            # We got all performance data
            # Make offloading decision
            # Make offloading calls

            offloading_decisions = cl_algorithm(federator_state.performance_data, {})

            federator_state.logger.info(f'Client ids are {client_ids}')
            federator_state.logger.info(f'Performance data is {federator_state.performance_data}')
            # pm_values = [x['pm'] for x in federator_state.performance_data.values()]


            # federator_state.logger.info(f'list {list(pm_values)}')
            # federator_state.logger.info(f'Mean compute time is {np.mean(list(pm_values))}')
            federator_state.logger.info(f'Starting offload')
            # offloading_decision = client_matching(federator_state.selected_clients, federator_state.performance_data)
            # federator_state.logger.info(f'Offloading decision {offloading_decision}')

            profiling_file = federator_state.config.output_path / f'profiling-{time.time()}.yaml'
            with open(profiling_file, 'w+') as file:
                perf_data = copy.deepcopy(federator_state.performance_data)


                for key, val in perf_data.items():
                    perf_data[key]['pm'] = np.array(list(val['pm'])).tolist()
                perf_data['offloading-decisions'] = copy.deepcopy(offloading_decisions)
                yaml.dump(perf_data, file)
                federator_state.logger.info('Perf data:')
                print(perf_data)
                federator_state.logger.info(f'Performance file at: {profiling_file}')

            for decision in offloading_decisions:
                federator_state.logger.info(f'Client {decision["from"]} will offload to client {decision["to"]}')
                federator_state.message_async(decision['from'], 'receive_offloading_decision', decision['to'], decision['when'])
                federator_state.message_async(decision['from'], 'unlock')

            # for c1, c2 in offloading_decision:
            #     federator_state.logger.info(f'Client {c1} will offload to client {c2}')
            #     federator_state.message_async(c1, 'receive_offloading_decision', c2, 0)
            #     federator_state.message_async(c1, 'unlock')

            alg_state['inactive'] = True

        return False

    def hook_post_training(self, federator_state, alg_state: dict):
        # Reset state for next round
        federator_state.algorithm_state = {}
        federator_state.logger.info("Resetting the algorithm state")