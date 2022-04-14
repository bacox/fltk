from abc import ABC, abstractmethod

from util.config import Config


class FederatedAlgorithm(ABC):
    name: str = 'Base'

    @abstractmethod
    def init_alg(self, federator_state, alg_state: dict, config: Config):
        '''
        Function that is called when the algorithm is initialized.
        :return:
        '''
        pass

    @abstractmethod
    def hook_post_startup(self, federator_state, alg_state: dict):
        '''
        Hook called tight after the system has started up.
        This means that all rpc communication is running
        and the (training) data is loaded on the clients
        :return:
        '''
        pass

    @abstractmethod
    def hook_client_selection(self, federator_state, alg_state: dict, round_id: int):
        '''
        Hook called before the client selection
        :return:
        '''
        pass

    @abstractmethod
    def hook_post_eval(self, federator_state, alg_state: dict, test_accuracy):
        '''
        Hook called after the evaluation on the Federator is done
        :return:
        '''
        pass

    @abstractmethod
    def hook_training(self, federator_state, alg_state: dict, deadline, training_start_time) -> bool:
        '''
        Hook call during the training loop
        :return:
        '''
        pass

    @abstractmethod
    def hook_post_training(self, federator_state, alg_state: dict):
        pass
