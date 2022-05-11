import copy
import time
import torch
import yaml
from fltk.strategy.algorithms.Alg import FederatedAlgorithm
from fltk.util.config import Config
from typing import List, Tuple
from scipy.stats import wasserstein_distance
import numpy as np
from itertools import zip_longest
from fltk.nets import get_net_split_point, get_net_feature_layers_names
from fltk.strategy.aggregation.FedAvg import fed_avg


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


def alter_dd_data(profiling_data: dict):
    max_value = max([max(x['dd']) for x in profiling_data.values()])
    for k, v in profiling_data.items():
        new_dd = np.zeros(max_value+1)
        for c, num in v['dd'].items():
            new_dd[c] = num
        v['dd'] = new_dd
    return profiling_data


def construct_similarity_matrix(performance_data: dict):
    performance_data = alter_dd_data(performance_data)
    for k, v in performance_data.items():
        performance_data[k]['ndd'] = normalize_class_distribution(v['dd'])
    return calc_similarity_matrix(performance_data)



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
    compute_times = []
    for x in perf_data.values():
        compute_times.append(node_complete_time(x))

    # compute_times = [sum(x['pm']) * (x['rl'] + x['ps']) for x in perf_data.values()]
    return np.mean(compute_times)


def node_complete_time(client_perf_data: dict):
    return sum(client_perf_data['pm']) * (client_perf_data['rl'] + client_perf_data['ps'])

def calc_decision_completion_time(slow_id, fast_id, performance_data: dict, offloading_point: int) -> Tuple[int, int]:
    client_slow = performance_data[slow_id]
    client_fast = performance_data[fast_id]
    def c_s(delta):
        return (client_slow['rl'] - delta) * client_slow['cut'] + (delta * client_slow['rcut'])

    def c_f(delta):
        return (client_fast['rl'] * client_fast['cut']) + (delta * client_fast['cut'])
    return c_s(offloading_point), c_f(offloading_point)

def cl_cost_function(decisions: List[dict], unused_clients: List[str], performance_data: dict, similarity_matrix: np.ndarray, offloading_sim_factor: float):
    # completion_times = [x for x in decisions.values()]
    completion_times = []
    calc_c_times = []
    clients = list(performance_data.keys())
    for d in decisions:
        c_s, c_f = calc_decision_completion_time(d['from'], d['to'], performance_data, d['when'])

        # print(f'Finding d["from"]={d["from"]} and d["to"]={d["to"]} in {clients}')
        # print(similarity_matrix)
        similarity = max(similarity_matrix[clients.index(d['from'])][clients.index(d['to'])], 0.01)
        # similarity = 0.01
        completion_times.append([[c_s, c_f], similarity])
        calc_c_times.append(max(c_s, c_f) * similarity)
    for unused_client in unused_clients:
        u_client = performance_data[unused_client]
        ect = u_client['rl'] * u_client['cut']
        similarity = 0.01

        completion_times.append([[ect],similarity])
        calc_c_times.append(ect* similarity)
    # Use list comprehension to flatmap the lists
    c_times = [c_time for c_times in completion_times for c_time in c_times[0]]
    # completion_times = [x[0] for x in completion_times]
    similarities = [x[1] for x in completion_times]
    return max(c_times) * max(offloading_sim_factor * sum(similarities), 0.01)


def generate_client_combinations(slow_ids: List[str], fast_ids: List[str]):
    import itertools
    if len(slow_ids) > len(fast_ids):
        return [list(zip(x, fast_ids)) for x in itertools.permutations(slow_ids, len(fast_ids))]
    else:
        return [list(zip(slow_ids, x)) for x in itertools.permutations(fast_ids, len(slow_ids))]


def find_offloading_point(client_slow: dict, client_fast: dict) -> int:
    def c_s(delta):
        return (client_slow['rl'] - delta) * client_slow['cut'] + (delta * client_slow['rcut'])
    def c_f(delta):
        return (client_fast['rl'] * client_fast['cut']) + (delta * client_fast['cut'])
    prev_cost = 0
    for d in range(client_slow['rl']):

        cost = max(c_s(d), c_f(d))
        if prev_cost and cost > prev_cost:
            # print(f'Found best cost at Delta ={d} with cost={cost}')
            return prev_cost
        prev_cost = cost
        # print(f'Delta ={d} with cost={cost}')
    return prev_cost

def generate_decision(combinations, performance_data: dict):
    all_client_ids = [x['id'] for x in performance_data.values()]
    decisions = []
    for slow_id, fast_id in combinations:
        all_client_ids.remove(slow_id)
        all_client_ids.remove(fast_id)
        # print(f'Offload from {slow_id} to {fast_id}')
        offloading_point = find_offloading_point(performance_data[slow_id], performance_data[fast_id])
        # slow_client = performance_data[slow_id]
        # fast_client = performance_data[fast_id]

        decision_id = f"{slow_id}-{fast_id}"
        decisions.append({
            'id': decision_id,
            'from': slow_id,
            'to': fast_id,
            'when': offloading_point,
            'response_id_to': f'O-{decision_id}-s',
            'expected_complete_time': {
                slow_id: 0,
                fast_id: 0
            }
            # 'response_id_from': f'O-{decision_id}-w'
        })
    return decisions, all_client_ids


def cl_algorithm(performance_data: dict, similarity_matrix: np.ndarray, offloading_sim_factor: float = 1):
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
    print(f'Starting offloading cl_algorithm with offloading_sim_factor={offloading_sim_factor}')
    # Mean compute time:
    mct = _mean_compute_time(performance_data)
    for key, item in performance_data.items():
        performance_data[key] = calc_compute_times(item, key)
    slow_clients = [v for k,v in performance_data.items() if sum(v['pm']) * (v['rl'] + v['ps']) > mct]
    fast_clients = [v for k,v in performance_data.items() if sum(v['pm']) * (v['rl'] + v['ps']) <= mct]

    slow_ids = [x['id'] for x in slow_clients]
    fast_ids = [x['id'] for x in fast_clients]
    print(f'Slow clients are: {slow_ids}')
    print(f'Fast clients are: {fast_ids}')
    # slow_ids
    # = ['4', '5']
    client_combinations = generate_client_combinations(slow_ids, fast_ids)
    potential_decisions_sets = []
    for c in client_combinations:
        decisions, unused_clients = generate_decision(c, performance_data)
        decision_set_cost = cl_cost_function(decisions, unused_clients, performance_data, similarity_matrix, offloading_sim_factor)
        potential_decisions_sets.append([decision_set_cost, decisions, unused_clients])

    print(f'Number of available decisions: {len(potential_decisions_sets)}')
    sorted_decisions_sets = sorted(potential_decisions_sets, key=lambda x: x[0])
    sorted_slow = sort_clients(slow_clients, reduced=True)
    sorted_fast = sort_clients(fast_clients, reduced=False)
    # min_matches = min(len(sorted_slow), len(sorted_fast))
    best_decision = sorted_decisions_sets[0]

    # decisions = []
    #
    # unsused_clients = []
    # # For now stick with the reverse sorted matching
    # for item in zip_longest(sorted_slow, sorted_fast):
    #     if not None in item:
    #         decision_id = f"{item[0]['id']}-{item[1]['id']}"
    #         decisions.append({
    #             'id': decision_id,
    #             'from': item[0]['id'],
    #             'to': item[1]['id'],
    #             'when': 0,
    #             'response_id_to': f'O-{decision_id}-s',
    #             'expected_complete_time': {
    #                 item[0]['id']: 0,
    #                 item[1]['id']: 0
    #             }
    #             # 'response_id_from': f'O-{decision_id}-w'
    #         })
    #     elif item[0] is None:
    #         unsused_clients.append(item[1]['id'])
    #     else:
    #         # item[1] is none
    #         unsused_clients.append(item[0]['id'])
    #
    #
    # cost = cl_cost_function(decisions, unsused_clients, performance_data, similarity_matrix)
    # print(f'The cost of these decisions is {cost}')
    return best_decision[1]



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

    def hook_training(self, federator_state, alg_state: dict, training_start_time, round_id) -> bool:
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
            sim_matrix = construct_similarity_matrix(federator_state.performance_data)
            offloading_decisions = cl_algorithm(federator_state.performance_data, sim_matrix, federator_state.config.offloading_similarity_factor)
            alg_state['offloading_decisions'] = offloading_decisions
            federator_state.logger.info(f'Client ids are {client_ids}')
            # federator_state.logger.info(f'Performance data is {federator_state.performance_data}')
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
                    perf_data[key]['rcut'] = val['rcut'].item()
                    perf_data[key]['rect'] = val['rect'].item()
                    perf_data[key]['rf'] = val['rf'].item()
                    perf_data[key]['ect'] = val['ect'].item()
                    perf_data[key]['cut'] = val['cut'].item()
                perf_data['offloading-decisions'] = copy.deepcopy(offloading_decisions)
                yaml.dump(perf_data, file)
                # federator_state.logger.info('Perf data:')
                # print(perf_data)
                federator_state.logger.info(f'Performance file at: {profiling_file}')
            unreleased_client_ids = client_ids
            for decision in offloading_decisions:
                unreleased_client_ids.remove(decision["to"])
                unreleased_client_ids.remove(decision["from"])
                decision['response_id_to'] = f"{round_id}-" + decision['response_id_to']
                # decision['response_id_from'] = f"{round_id}-" + decision['response_id_from']
                federator_state.logger.info(f'Client {decision["from"]} will offload to client {decision["to"]}')
                federator_state.create_response_expectation(decision['response_id_to'])
                # federator_state.create_response_expectation(decision['response_id_from'])
                federator_state.message_async(decision['from'], 'receive_offloading_decision', decision['to'], decision['when'], decision['response_id_to'])
                federator_state.message_async(decision['from'], 'unlock')

            # for c1, c2 in offloading_decision:
            #     federator_state.logger.info(f'Client {c1} will offload to client {c2}')
            #     federator_state.message_async(c1, 'receive_offloading_decision', c2, 0)
            #     federator_state.message_async(c1, 'unlock')
            for unreleased_id in unreleased_client_ids:
                federator_state.logger.info(f'Client {unreleased_id} not used for offloading -> Unlocking')
                federator_state.message_async(unreleased_id, 'unlock')
            alg_state['inactive'] = True

        return False

    def hook_post_training(self, federator_state, alg_state: dict):
        # Reset state for next round
        federator_state.algorithm_state = {}
        federator_state.logger.info("Resetting the algorithm state")

    def pre_agggrate_merge(self, weights_a, num_samples_a, weights_b, num_samples_b):
        weights = {'a': weights_a, 'b': weights_b}
        sizes = {'a': num_samples_a, 'b':num_samples_b}
        return fed_avg(weights, sizes), np.max([num_samples_a,num_samples_b])

    def pre_aggregate_glue(self, model_a_data, model_b_data, feature_layer_names: List[str], split_point: int):
        '''

        :param model_a_data: To original model
        :param model_b_data: The offloaded model.
        :param feature_layer_names:
        :param split_point:
        :return:
        '''
        print(f'Glue-ing to models from layer {split_point} or names {feature_layer_names}')

        for name in model_a_data.keys():
            if any([True for x in feature_layer_names if str(name).startswith(x)]):
                model_a_data[name].data += model_b_data[name].data
                print(f'Matching {name} on {feature_layer_names}')
        return model_a_data


    def hook_pre_aggregation(self, federator_state, alg_state: dict, round_id: int):
        # Call super to filter responses based on the current round_id
        super().hook_pre_aggregation(federator_state, alg_state, round_id)

        federator_state.logger.info(f'Found following responses')
        for k,_ in federator_state.response_store.items():
            federator_state.logger.info(f'-\t {k}')
        to_merge = []
        offloaded = [k for k, v in federator_state.response_store.items() if k.startswith(f'{round_id}-O-')]
        federator_state.logger.info(f'Found the following offloaded models: {offloaded}')
        net_split_point = get_net_split_point(federator_state.config.net_name)
        feature_layer_names = get_net_feature_layers_names(federator_state.config.net_name)
        for offloaded_client in offloaded:
            decision_id = '-'.join(offloaded_client.split("-")[2:4])


            # alg_state
            to_match = f'{round_id}-{offloaded_client.split("-")[2]}'
            decision = [x for x in alg_state['offloading_decisions'] if x['id'] == decision_id]
            federator_state.logger.info(f'Need to merge {offloaded_client} and {to_match} with decision id: {decision}')
            # responses: dict = federator_state.response_store
            resp_model_b = federator_state.response_store.pop(offloaded_client)
            resp_model_a = federator_state.response_store[to_match]
            # resp_data_a = resp_model_a['response_data']
            _, weights_a, _, _, _, _, _, num_samples_a = resp_model_a['response_data']
            if "response_data" in resp_model_b:
                _, weights_b, _, _, _, _, _, num_samples_b = resp_model_b['response_data']
            else:
                federator_state.logger.info(
                    f'Cannot find response data in {offloaded_client}; Using data from weak node only')
                federator_state.response_store[to_match]['response_data'][1] = weights_a
                federator_state.response_store[to_match]['response_data'][7] = num_samples_a
                continue
                # _, weights_b, _, _, _, _, _, num_samples_b = resp_model_a['response_data']
            print(f'Merging A: {weights_a.keys()}')
            print(f'With keys B: {weights_b.keys()}')

            offloading_choice = federator_state.config.offloading_option
            if offloading_choice == 2:
                # Last options (Combined)
                merged_weights, merged_num_samples = self.pre_agggrate_merge(weights_a, num_samples_a, weights_b,
                                                                             num_samples_b)
                merged_weights = self.pre_aggregate_glue(merged_weights, weights_b, feature_layer_names, net_split_point)
                merged_num_samples = np.max([num_samples_a, num_samples_b])
            elif offloading_choice == 1:
                # Glue models together (Option 2)
                merged_weights = self.pre_aggregate_glue(weights_a, weights_b, feature_layer_names, net_split_point)
                merged_num_samples = np.max([num_samples_a, num_samples_b])
            else:
                # Merge using fedavg (Option 1)
                merged_weights, merged_num_samples = self.pre_agggrate_merge(weights_a, num_samples_a, weights_b,
                                                                             num_samples_b)

            federator_state.response_store[to_match]['response_data'][1] = merged_weights
            federator_state.response_store[to_match]['response_data'][7] = merged_num_samples
