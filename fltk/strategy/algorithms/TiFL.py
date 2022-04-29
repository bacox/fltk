from fltk.core.client import Client
from fltk.strategy.algorithms.Alg import FederatedAlgorithm
from fltk.strategy.client_selection.tifl import create_tiers, tier_selection
from fltk.util.config import Config


class TiFL(FederatedAlgorithm):
    name: str = 'tifl'

    def __init__(self):
        self.n_tiers = None
        self.I = None

    def init_alg(self, federator_state, alg_state: dict, config: Config):
        self.I = config.tifl_I
        self.n_tiers = config.tifl_n_tiers
        if self.I is None:
            self.I = int(self.n_tiers)

    def hook_post_startup(self, federator_state, alg_state: dict):
        # federator_state.client_profiling()
        training_futures = []
        num_local_updates = 300
        for client in federator_state.clients:
            fut = [client.name, federator_state.message_async(client.ref, Client.profile_offline, num_local_updates)]
            training_futures.append(fut)
        profiling_data = []
        for fut in training_futures:
            profiling_data.append((fut[0], fut[1].wait()))
        tier_data = create_tiers(profiling_data, 3, federator_state.config.rounds)
        alg_state['tiers'] = tier_data

    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        alg_state['tiers'], selected_tier = tier_selection(alg_state['tiers'], round_id, self.I)
        alg_state['selected_tier_id'] = selected_tier.id
        federator_state.client_pool = [x for x in federator_state.clients if x.name in selected_tier.client_ids]

    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        selected_tier = [x for x in alg_state['tiers'] if x.id == alg_state['selected_tier_id']][0]
        selected_tier.accuracy = test_accuracy
        federator_state.logger.info(f'After test eval tier data-> {alg_state}')

    def hook_training(self, federator_state, alg_state: dict, training_start_time) -> bool:
        pass

    def hook_post_training(self, federator_state, alg_state: dict):
        pass
