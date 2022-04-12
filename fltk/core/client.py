import time
from typing import Tuple, Any

import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import Config
from fltk.util.profilerV3 import Profiler
from fltk.nets import get_net_split_point
import numpy as np


class Client(Node):
    running = False
    terminate_training = False
    locked = False
    has_offloading_request = False

    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        super().__init__(id, rank, world_size, config)
        self.offloading_decision = {}
        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.nets.selected().parameters(),
                                                   **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.logger, self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

    def remote_registration(self):
        self.logger.info('Sending registration')
        self.message('federator', 'register_client', self.id, self.rank)
        self.running = True
        self._event_loop()

    def offload_call(self, offload_client_id):
        pass

    def lock(self):
        self.logger.info(f'{self.id} is locked')
        self.locked = True

    def unlock(self):
        self.logger.info(f'Unlocking client {self.id}')
        self.locked = False

    def is_locked(self):
        return self.locked

    def stop_client(self):
        self.logger.info('Got call to stop event loop')
        self.running = False

    def _event_loop(self):
        self.logger.info('Starting event loop')
        while self.running:
            time.sleep(0.1)
        self.logger.info('Exiting node')

    def train(self, num_epochs: int, use_profiler = True):
        if not self.real_time:
            use_profiler = False
        start_time = time.time()

        running_loss = 0.0
        final_running_loss = 0.0
        if self.distributed:
            self.dataset.train_sampler.set_epoch(num_epochs)

        has_send_metric = False
        number_of_training_samples = len(self.dataset.get_train_loader())

        # Init profiler
        net_split_point = get_net_split_point(self.config.net_name)
        profiling_size = 100
        p_data = np.zeros(profiling_size)
        iter = 0
        p = Profiler(profiling_size, net_split_point - 1, use_profiler)
        network = self.nets.selected()
        if p.active:
            p.attach(network)
        # self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')

        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            s_time = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            if p.active:
                p.signal_forward_start()

            # Calculate prediction
            outputs = network(inputs)
            # Determine loss
            loss = self.loss_function(outputs, labels)

            if p.active:
                # p.signal_backward_start()
                p.signal_forward_end()
                p.signal_backwards_start()
            # Correct for errors
            loss.backward()
            self.optimizer.step()
            if p.active:
                p.signal_backwards_end()
                p.step()

            running_loss += loss.item()
            e_time = time.time() - s_time
            # This is here for debugging. can be removed later
            if p.active:
                p_data[iter] = e_time
                iter += 1

            # Mark logging update step
            if i % self.config.log_interval == 0:
                self.logger.info(
                    '[%s] [%d, %5d] loss: %.3f' % (self.id, num_epochs, i, running_loss / self.config.log_interval))
                final_running_loss = running_loss / self.config.log_interval
                running_loss = 0.0

            if p.active:
                if i == profiling_size - 1:
                    p.active = False
                    p.remove_all_handles()
                    profiler_data = p.aggregate_values()
                    self.logger.info(f'Profiler data: {profiler_data}')
                    self.logger.info(f'Profiler data (sum): {np.sum(profiler_data)} and {e_time} and {np.mean(p_data)}')
                    self.logger.info(f'Profiler data (%): {np.abs(np.mean(p_data) - np.sum(profiler_data)) / np.sum(profiler_data)}')
                    self.message_async('federator', 'save_performance_metric', self.id, profiler_data)

            if self.terminate_training:
                break

            if self.offloading_decision:
                # Do not check when for now, just execute
                self.logger.info(f'{self.id} is offloading to {self.offloading_decision["node-id"]}')
                self.message_async(self.offloading_decision['node-id'], Client.receive_offloading_request, self.id, self.get_nn_parameters())
                self.freeze_layers(network, net_split_point)
                self.message_async(self.offloading_decision['node-id'], Client.unlock)
                self.offloading_decision = {}


        end_time = time.time()
        duration = end_time - start_time
        # self.logger.info(f'Train duration is {duration} seconds')
        self.unfreeze_layers()
        return final_running_loss, self.get_nn_parameters(),

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()
        E = self.config.epochs
        B = 16  # nicely hardcoded :)
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self):
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        network = self.nets.selected()
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                if self.terminate_training:
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = network(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        loss /= len(self.dataset.get_test_loader().dataset)
        if total:
            accuracy = 100.0 * correct / total
        else:
            accuracy = 0
        # confusion_mat = confusion_matrix(targets_, pred_)
        # accuracy_per_class = confusion_mat.diagonal() / confusion_mat.sum(1)
        #
        # class_precision = calculate_class_precision(confusion_mat)
        # class_recall = calculate_class_recall(confusion_mat)
        end_time = time.time()
        duration = end_time - start_time
        # self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss

    def get_client_datasize(self):
        return len(self.dataset.get_train_sampler())

    def receive_offloading_request(self, sender_id, model_params):
        self.has_offloading_request = True
        # Initialize net instead of just copying model params
        self.nets[sender_id] = model_params

    def receive_offloading_decision(self, node_id, when: float):
        self.offloading_decision['node-id'] = node_id
        self.offloading_decision['when'] = when

    def stop_training(self):
        self.terminate_training = True
        self.logger.info('Got a call to stop training')

    def exec_round(self, num_epochs: int) -> Tuple[Any, Any, Any, Any, float, float, float]:
        start = time.time()
        self.terminate_training = False
        loss, weights = self.train(num_epochs)
        time_mark_between = time.time()
        accuracy, test_loss = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        # self.logger.info(f'Round duration is {duration} seconds')

        while self.is_locked():
            time.sleep(0.1)

        if self.has_offloading_request:
            other_client_id = self.nets.other_keys()[0]
            self.logger.info(f'I need to train the offloading model from client {other_client_id} as well!')
            self.nets.select(other_client_id)
            loss, weights = self.train(num_epochs, False)
            # time_mark_between = time.time()
            accuracy, test_loss = self.test()
            self.logger.info('Stopping training of alternative model')
            self.nets.reset()
            trained_offloaded_model = self.nets.remove_model(other_client_id)


        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, v in weights.items():
            weights[k] = v.cpu()
        self.logger.info('Ending training')
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')