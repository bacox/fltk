from strategy.algorithms.Alg import FederatedAlgorithm
from util.config import Config


class Vanilla(FederatedAlgorithm):
    name: str = 'vanilla'

    def init_alg(self, federator_state, alg_state: dict, config: Config):
        pass

    def hook_post_startup(self, federator_state, alg_state: dict):
        pass

    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        # Use all the clients for the client selection
        federator_state.client_pool = federator_state.clients

    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        pass

    def hook_training(self, federator_state, alg_state: dict, deadline, training_start_time) -> bool:
        pass

    def hook_post_training(self, federator_state, alg_state: dict):
        pass
