import copy
import os
from typing import Callable, Any
import torch
from torch.distributed import rpc
from fltk.datasets.loader_util import get_dataset
from fltk.nets import get_net_by_name
from fltk.util.config import Config
from fltk.util.log import getLogger

# Global dictionary to enable peer to peer communication between clients
global_vars = {}


def _remote_method_direct(method, other_node: str, *args, **kwargs):
    """
    Utility function for RPC communication between nodes
    :param method: A callable
    :param other_node: reference to other node
    :return: any
    """
    args = [method, other_node] + list(args)
    return rpc.rpc_sync(other_node, method, args=args, kwargs=kwargs)


class Nets(dict):
    __default__ = '__default__'

    def __init__(self) -> None:
        super().__init__()
        self.default = self.__default__

    def select(self, key: str):
        self.default = key

    def reset(self):
        self.default = self.__default__

    def selected(self):
        return self.get(self.default)

    def remove_model(self, key):
        self.pop(key)

    def other_keys(self):
        return [x for x in self.keys() if x != self.__default__]


class Node:
    """
    Implementation of any participating node.
    It handles communication and the basic functions for Deep Learning.
    """
    id: int
    rank: int
    world_size: int
    real_time: bool = False
    distributed: bool = True
    cuda: bool = False
    finished_init: bool = False
    # net: Any
    nets: Nets
    dataset: Any
    logger = getLogger(__name__)

    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        self.config = config
        self.device = torch.device("cpu")
        self.id = id
        self.rank = rank
        self.world_size = world_size
        self.real_time = config.real_time
        global global_vars
        global_vars['self'] = self
        self.nets = Nets()
        self._config(config)

    def _config(self, config: Config):
        self.logger.setLevel(config.log_level.value)
        self.config.rank = self.rank
        self.config.world_size = self.world_size
        self.cuda = config.cuda
        self.device = self.init_device()
        self.distributed = config.distributed
        self.set_net(self.load_default_model())

    def init_dataloader(self, world_size: int = None):
        config = copy.deepcopy(self.config)
        if world_size:
            config.world_size = world_size
        self.logger.info(f'world size = {config.world_size} with rank={config.rank}')
        self.dataset = get_dataset(config.dataset_name)(config)
        self.finished_init = True
        self.logger.info('Done with init')

    def is_ready(self):
        return self.finished_init

    @staticmethod
    def _receive(method: Callable, sender: str, *args, **kwargs):
        """
        Communication utility function.
        This is the entry points for all incoming RPC communication.
        The class object (self) will be loaded from the global space
        and the callable method is executed within the context of self
        :param method:
        :param sender:
        :param args:
        :param kwargs:
        :return:
        """
        global global_vars
        global_self = global_vars['self']
        if type(method) is str:
            method = getattr(global_self, method)
            return method(*args, **kwargs)
        else:
            return method(global_self, *args, **kwargs)

    def init_device(self):
        if self.cuda and not torch.cuda.is_available():
            self.logger.warning('Unable to configure device for GPU because cuda.is_available() == False')
        if self.cuda and torch.cuda.is_available():
            self.logger.info("Configure device for GPU (Cuda)")
            return torch.device("cuda:0")
        else:
            self.logger.info("Configure device for CPU")
            return torch.device("cpu")

    def set_net(self, net, key: str = '__default__'):
        """
        Update the local parameters of self.net with net.
        This method also makes sure that the parameters are configured for the correct device (CPU or GPU/CUDA)
        :param net:
        """
        self.nets[key] = net
        # self.net = net
        self.nets[key].to(self.device)

    def get_nn_parameters(self, key: str = '__default__'):
        """
        Return the DNN parameters.
        """
        return self.nets[key].state_dict()

    def load_default_model(self):
        """
        Load a model from default model file.
        This is used to ensure consistent default model behavior.
        """
        model_class = get_net_by_name(self.config.net_name)
        default_model_path = os.path.join(self.config.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.
        :param model_file_path: string
        """
        model_class = get_net_by_name(self.config.net_name)
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.logger.warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.logger.warning("Could not find model: {}".format(model_file_path))
        return model


    def update_nn_parameters(self, new_params, key:str = '__default__'):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.nets[key].load_state_dict(copy.deepcopy(new_params), strict=True)

    def message(self, other_node: str, method: Callable, *args, **kwargs) -> torch.Future:
        """
        All communication with other nodes should go through this method.
        The attribute real_time determines if the communication should use RPC or if it is a direct object call.
        :return: (resolved) torch.Future
        """
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_sync(other_node, func, args=args_list,  kwargs=kwargs)
        return method(other_node, *args, **kwargs)

    def message_async(self, other_node: str, method: Callable, *args, **kwargs) -> torch.Future:
        """
        This is the async version of 'message'.
        All communication with other nodes should go through this method.
        The attribute real_time determines if the communication should use RPC or if it is a direct object call.
        :return: torch.Future
        """
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_async(other_node, func, args=args_list,  kwargs=kwargs)
        # Wrap inside a future to keep the logic the same
        future = torch.futures.Future()
        future.set_result(method(other_node, *args, **kwargs))
        return future

    def freeze_layers(self, net, until: int):
        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flatt_children = []
            if children == []:
                return model
            else:
                for child in children:
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children

        for idx, layer in enumerate(get_children(net)):
            if idx < until:
                self.logger.debug(f'[{idx}] Freezing layer: {layer}')
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_layers(self, key: str = '__default__'):
        for param in self.nets[key].parameters():
            param.requires_grad = True

    def ping(self, sender: str):
        """
        Utility function that can be used to test the connectivity between nodes.
        :param sender: str
        :return: str
        """
        self.logger.info(f'{self.id} got a ping from {sender}')
        return 'Pong'

    def __repr__(self):
        return str(self.id)
