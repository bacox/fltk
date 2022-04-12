import copy
import time
from pathlib import Path
from typing import List, Union

import torch
from tqdm import tqdm
import numpy as np
from fltk.core.client import Client
from fltk.core.node import Node
from fltk.datasets.loader_util import get_dataset
from fltk.strategy import FedAvg, random_selection, average_nn_parameters, average_nn_parameters_simple
from fltk.util.config import Config
from dataclasses import dataclass

from fltk.util.data_container import DataContainer, FederatorRecord, ClientRecord
from fltk.strategy import get_aggregation
from fltk.strategy.algorithms.deadline import deadline_callable
from fltk.strategy.algorithms.offloading import offloading_callable

NodeReference = Union[Node, str]
@dataclass
class LocalClient:
    name: str
    ref: NodeReference
    data_size: int
    exp_data: DataContainer
    valid_response: bool = True
    available: bool = True


def cb_factory(future: torch.Future, method, *args, **kwargs):
    future.then(lambda x: method(x, *args, **kwargs))

class Federator(Node):
    clients: List[LocalClient] = []
    # clients: List[NodeReference] = []
    num_rounds: int
    exp_data: DataContainer
    callables = {
        deadline_callable: {'active': False, 'state': {}},
        offloading_callable: {'active': False, 'state': {}}
    }
    # callables = [
    #     {'callable': deadline_callable, 'active': False, 'state': {}},
    #     {'callable': offloading_callable, 'active': True, 'state': {}}
    # ]

    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        super().__init__(id, rank, world_size, config)
        self.loss_function = self.config.get_loss_function()()
        self.num_rounds = config.rounds
        self.config = config
        prefix_text = ''
        if config.replication_id:
            prefix_text = f'_r{config.replication_id}'
        config.output_path = Path(config.output_path) / f'{config.experiment_prefix}{prefix_text}'
        self.exp_data = DataContainer('federator', config.output_path, FederatorRecord, config.save_data_append)
        Config.ToYamlFile(config, config.output_path / 'config.yaml')
        self.aggregation_method = get_aggregation(config.aggregation)
        self.selected_clients: List[LocalClient] = []
        self.performance_data = {}



    def create_clients(self):
        self.logger.info('Creating clients')
        if self.config.single_machine:
            # Create direct clients
            world_size = self.config.num_clients + 1
            for client_id in range(1, self.config.num_clients+ 1):
                client_name = f'client{client_id}'
                client = Client(client_name, client_id, world_size, copy.deepcopy(self.config))
                self.clients.append(LocalClient(client_name, client, 0, DataContainer(client_name, self.config.output_path,
                                                                                      ClientRecord, self.config.save_data_append)))
                self.logger.info(f'Client "{client_name}" created')

    def register_client(self, client_name, rank):
        self.logger.info(f'Got new client registration from client {client_name}')
        if self.config.single_machine:
            self.logger.warning('This function should not be called when in single machine mode!')
        self.clients.append(LocalClient(client_name, client_name, rank, DataContainer(client_name, self.config.output_path,
                                                                                      ClientRecord, self.config.save_data_append)))

    def stop_all_clients(self):
        for client in self.clients:
            self.message(client.ref, Client.stop_client)

    def save_performance_metric(self, client_id, metric):
        self.performance_data[client_id] = metric

    def _num_clients_online(self) -> int:
        return len(self.clients)

    def _all_clients_online(self) -> bool:
        return len(self.clients) == self.world_size - 1

    def clients_ready(self):
        """
        Synchronous implementation
        """
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            all_ready = True
            for client in self.clients:
                resp = self.message(client.ref, Client.is_ready)
                if resp:
                    self.logger.info(f'Client {client} is ready')
                else:
                    self.logger.info(f'Waiting for client {client}')
                    all_ready = False
            time.sleep(2)

    def get_client_data_sizes(self):
        for client in self.clients:
            client.data_size = self.message(client.ref, Client.get_client_datasize)

    def run(self):
        # Load dataset with world size 2 to load the whole dataset.
        # Caused by the fact that the dataloader subtracts 1 from the world size to exclude the federator by default.
        self.init_dataloader(world_size=2)

        self.create_clients()
        while not self._all_clients_online():
            self.logger.info(f'Waiting for all clients to come online. Waiting for {self.world_size - 1 -self._num_clients_online()} clients')
            time.sleep(2)
        self.logger.info('All clients are online')
        # self.logger.info('Running')
        # time.sleep(10)
        self.client_load_data()
        self.get_client_data_sizes()
        self.clients_ready()
        # self.logger.info('Sleeping before starting communication')
        # time.sleep(20)
        for communication_round in range(self.config.rounds):
            self.exec_round(communication_round)

        self.save_data()
        self.logger.info('Federator is stopping')


    def save_data(self):
        self.exp_data.save()
        for client in self.clients:
            client.exp_data.save()

    def client_load_data(self):
        for client in self.clients:
            self.message(client.ref, Client.init_dataloader)

    def set_tau_eff(self):
        total = sum(client.data_size for client in self.clients)
        # responses = []
        for client in self.clients:
            self.message(client.ref, Client.set_tau_eff, client.ref, total)
            # responses.append((client, _remote_method_async(Client.set_tau_eff, client.ref, total)))
        # torch.futures.wait_all([x[1] for x in responses])

    def test(self, net):
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total
        # confusion_mat = confusion_matrix(targets_, pred_)
        # accuracy_per_class = confusion_mat.diagonal() / confusion_mat.sum(1)
        #
        # class_precision = calculate_class_precision(confusion_mat)
        # class_recall = calculate_class_recall(confusion_mat)
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss

    def exec_round(self, id: int):
        self.logger.info('='*20)
        self.logger.info(f'= Starting round {id} =')
        self.logger.info('='*20)
        start_time = time.time()
        num_epochs = self.config.epochs

        # Client selection
        self.selected_clients = random_selection(self.clients, self.config.clients_per_round)

        last_model = self.get_nn_parameters()
        for client in self.selected_clients:
            client.valid_response = True
            self.message(client.ref, Client.update_nn_parameters, last_model)

        client_weights = {}
        client_sizes = {}
        training_futures: List[torch.Future] = []

        def training_cb(fut: torch.Future, client_ref: LocalClient, client_weights, client_sizes, num_epochs):
            train_loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration = fut.wait()
            self.logger.info(f'Training callback for client {client_ref.name} with accuracy={accuracy}')
            if client_ref.valid_response:
                client_weights[client_ref.name] = weights
                client_data_size = self.message(client_ref.ref, Client.get_client_datasize)
                client_sizes[client_ref.name] = client_data_size
                client_ref.exp_data.append(
                    ClientRecord(id, train_duration, test_duration, round_duration, num_epochs, 0, accuracy, train_loss,
                                 test_loss))
            else:
                self.logger.info(f'Omitting client response because it is marked invalid!')

        for client in self.selected_clients:
            future = self.message_async(client.ref, Client.exec_round, num_epochs)
            cb_factory(future, training_cb, client, client_weights, client_sizes, num_epochs)
            self.logger.info(f'Request sent to client {client.name}')
            training_futures.append(future)

        def all_futures_done(futures: List[torch.Future]) -> bool:
            return all(map(lambda x: x.done(), futures))

        deadline = 3
        training_start_time = time.time()
        stop_loop = False
        while not all_futures_done(training_futures) and not stop_loop:
            for (c, c_data) in [(x, c_data) for x, c_data in self.callables.items() if c_data['active']]:
                stop_loop = c(self, c_data['state'], deadline, training_start_time)
                if stop_loop:
                    break
            time.sleep(0.1)

        if len(client_weights):
            updated_model = self.aggregation_method(client_weights, client_sizes)
            self.update_nn_parameters(updated_model)
        else:
            self.logger.warning(f'Skipping the aggregation step due to missing client weights! Number of client weights = {len(client_weights)}')
        test_accuracy, test_loss = self.test(self.nets.selected())
        self.logger.info(f'[Round {id:>3}] Federator has a accuracy of {test_accuracy} and loss={test_loss}')

        end_time = time.time()
        duration = end_time - start_time
        self.exp_data.append(FederatorRecord(len(self.selected_clients), id, duration, test_loss, test_accuracy))
        self.logger.info(f'[Round {id:>3}] Round duration is {duration} seconds')
        self.performance_data = {}
        for c_data in self.callables.values():
            c_data['state'] = {}
