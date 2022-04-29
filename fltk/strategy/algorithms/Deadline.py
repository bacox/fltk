import time
from fltk.core.client import Client
from fltk.strategy.algorithms.Alg import FederatedAlgorithm
from fltk.util.config import Config


class Deadline(FederatedAlgorithm):
    name: str = 'deadline'

    def __init__(self):
        self.deadline_time = None

    def init_alg(self, federator_state, alg_state: dict, config: Config):
        self.deadline_time = config.deadline_time

    def hook_post_startup(self, federator_state, alg_state: dict):
        pass

    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        pass

    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        pass

    def hook_training(self, federator_state, alg_state: dict, training_start_time) -> bool:
        if time.time() > training_start_time + self.deadline_time:
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
