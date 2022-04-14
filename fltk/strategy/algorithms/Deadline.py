import time
from core.client import Client
from strategy.algorithms.Alg import FederatedAlgorithm
from util.config import Config


class Deadline(FederatedAlgorithm):
    name: str = 'deadline'

    def init_alg(self, federator_state, alg_state: dict, config: Config):
        pass

    def hook_post_startup(self, federator_state, alg_state: dict):
        pass

    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        pass

    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        pass

    def hook_training(self, federator_state, alg_state: dict, deadline, training_start_time) -> bool:
        if time.time() > training_start_time + deadline:
            federator_state.logger.warning('Deadline has passed!')
            # Notify clients to stop
            for client in federator_state.selected_clients:
                client.valid_response = False
                federator_state.message_async(client.ref, Client.stop_training)
            # Break out waiting loop
            return True

    def hook_post_training(self, federator_state, alg_state: dict):
        # Reset state for next round
        alg_state = {}
