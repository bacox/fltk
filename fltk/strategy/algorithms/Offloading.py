from strategy.algorithms.Alg import FederatedAlgorithm
from util.config import Config
import numpy as np


def client_matching(client_data, performance_data):
    mct = np.mean(list(performance_data.values()))
    return [(client_data[0].name, client_data[1].name)]


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

    def hook_training(self, federator_state, alg_state: dict, deadline, training_start_time) -> bool:
        client_ids = [x.name for x in federator_state.selected_clients]
        if 'first_cycle' not in alg_state:
            for client_id in client_ids:
                federator_state.message_async(client_id, 'lock')
            alg_state['first_cycle'] = False

        if 'inactive' not in alg_state and all(item in federator_state.performance_data for item in client_ids):
            # We got all performance data
            # Make offloading decision
            # Make offloading calls
            federator_state.logger.info(f'Client ids are {client_ids}')
            federator_state.logger.info(f'Performance data is {federator_state.performance_data}')
            federator_state.logger.info(f'list {list(federator_state.performance_data.values())}')
            federator_state.logger.info(f'Mean compute time is {np.mean(list(federator_state.performance_data.values()))}')
            federator_state.logger.info(f'Starting offload')
            offloading_decision = client_matching(federator_state.selected_clients, federator_state.performance_data)
            federator_state.logger.info(f'Offloading decision {offloading_decision}')
            for c1, c2 in offloading_decision:
                federator_state.logger.info(f'Client {c1} will offload to client {c2}')
                federator_state.message_async(c1, 'receive_offloading_decision', c2, 0)
                federator_state.message_async(c1, 'unlock')

            alg_state['inactive'] = True

        return False

    def hook_post_training(self, federator_state, alg_state: dict):
        # Reset state for next round
        alg_state = {}
