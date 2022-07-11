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

    def train(self, round_id: int, num_epochs: int, use_profiler = True, limit_num_training_updates: int = 0):
        if not self.real_time:
            use_profiler = False
        # use_profiler = False
        start_time = time.time()

        running_loss = 0.0
        final_running_loss = 0.0

        has_send_metric = False
        number_of_training_samples = len(self.dataset.get_train_loader()) * num_epochs
        remaining_training_samples = number_of_training_samples
        # Init profiler
        net_split_point = get_net_split_point(self.config.net_name)
        profiling_size = 100
        p_data = np.zeros(profiling_size)
        iter = 0
        p = Profiler(profiling_size, net_split_point - 1, use_profiler)
        network = self.nets.selected()
        if p.active:
            p.attach(network)
        self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')
        i = 1
        for epoch in range(num_epochs):
            if self.distributed:
                self.dataset.train_sampler.set_epoch(round_id + epoch)
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

                remaining_training_samples -= self.config.batch_size
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.debug(
                        '[%s] [%d, %5d] loss: %.3f' % (self.id, round_id, i, running_loss / self.config.log_interval))
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0

                if limit_num_training_updates and i >= limit_num_training_updates:
                    break

                if p.active:
                    if i == profiling_size - 1:
                        p.active = False
                        p.remove_all_handles()
                        profiler_data = p.aggregate_values()
                        self.logger.info(f'Profiler data: {profiler_data}')
                        self.logger.info(f'{remaining_training_samples=}')

                        self.logger.info(f'Profiler data (sum): {np.sum(profiler_data)} and {e_time} and {np.mean(p_data)}')
                        self.logger.info(f'Profiler data (%): {np.abs(np.mean(p_data) - np.sum(profiler_data)) / np.sum(profiler_data)}')
                        profiling_obj = {
                            'pm': profiler_data,
                            'ps': profiling_size,
                            'bs': self.config.batch_size,
                            'rl': number_of_training_samples - i,
                            'dd': dict(zip(self.labels_dist[0].tolist(), self.labels_dist[1].tolist()))
                        }
                        self.message_async('federator', 'save_performance_metric', self.id, profiling_obj)

                if self.terminate_training:
                    break

                if self.offloading_decision:
                    # @TODO: Incoorporate offloading_reponse_id in message
                    # Do not check when for now, just execute
                    self.logger.info(f'{self.id} is offloading to {self.offloading_decision["node-id"]}')
                    rem_local_updates = number_of_training_samples - i
                    self.logger.info(f'REMAINING LOCAL UPDATES IS {number_of_training_samples} - {i} = {rem_local_updates}')
                    # def offloading_cb(fut):
                    #     fut.wait()
                    #     self.logger.info(f'Offloading request done -> Unlocking client {self.offloading_decision["node-id"]}')
                    #     self.message_async(self.offloading_decision['node-id'], Client.unlock)
                    # offloading_future = self.message_async(self.offloading_decision['node-id'], Client.receive_offloading_request, self.id, self.get_nn_parameters(), self.offloading_decision['response_id_to'], rem_local_updates)
                    # offloading_future.then(offloading_cb)
                    self.message(self.offloading_decision['node-id'], Client.receive_offloading_request, self.id, self.get_nn_parameters(), self.offloading_decision['response_id_to'], rem_local_updates)
                    # offloading_future.then(lambda x: self.message_async(self.offloading_decision['node-id'], Client.unlock))
                    self.freeze_layers(network, net_split_point)
                    self.logger.info(f'Offloading request done -> Unlocking client {self.offloading_decision["node-id"]}')
                    self.message_async(self.offloading_decision['node-id'], Client.unlock)
                    self.offloading_decision = {}


        end_time = time.time()
        duration = end_time - start_time
        # self.logger.info(f'Train duration is {duration} seconds')
        self.unfreeze_layers()
        return final_running_loss, self.get_nn_parameters(), i

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

    def profile_offline(self, num_updates):
        print(f'Start client profiling of {self.id}')
        start_time = time.time()
        number_of_training_samples = len(self.dataset.get_train_loader())
        network = self.nets.selected()
        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # Calculate prediction
            outputs = network(inputs)
            # Determine loss
            loss = self.loss_function(outputs, labels)
            # Correct for errors
            loss.backward()
            self.optimizer.step()
            if i >= num_updates:
                break
        end_time = time.time()
        duration = end_time - start_time
        num_updates = min(i, num_updates)
        print(f'Client {self.id}, ({duration=} / {num_updates=}) * {number_of_training_samples=}')
        estimated_full_duration = (duration / num_updates) * number_of_training_samples
        return estimated_full_duration

    def get_client_datasize(self):
        return len(self.dataset.get_train_sampler())

    def receive_offloading_request(self, sender_id, model_params, reponse_id: str, rem_local_updates: int):
        self.logger.info('Received offloading request')
        self.has_offloading_request = True
        self.offloading_reponse_id = reponse_id
        self.offloading_rem_local_updates = rem_local_updates
        # Initialize net instead of just copying model params
        # model_params
        self.set_net(self.load_default_model(), sender_id)
        self.update_nn_parameters(model_params, sender_id)
        # net = self.nets[sender_id]
        # self.nets[sender_id] = model_params

    def receive_offloading_decision(self, node_id, when: float, response_id_to):
        self.offloading_decision['node-id'] = node_id
        self.offloading_decision['when'] = when
        self.offloading_decision['response_id_to'] = response_id_to
        # self.offloading_decision['response_id_from'] = response_id_from

    def stop_training(self):
        self.terminate_training = True
        self.logger.info('Got a call to stop training')

    def exec_round(self, round_id: int, num_epochs: int, response_id: str, server_ref) -> Tuple[Any, Any, Any, Any, float, float, float]:
        start = time.time()
        self.terminate_training = False
        loss, weights, num_samples = self.train(round_id, num_epochs)
        time_mark_between = time.time()
        accuracy, test_loss = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.logger.info('Calling pre_communicate function')
            self.optimizer.pre_communicate()
        for k, v in weights.items():
            weights[k] = v.cpu()
        self.message_async(server_ref, 'receive_training_result', response_id, [loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, num_samples])
        # self.logger.info(f'Round duration is {duration} seconds')

        while self.is_locked():
            time.sleep(0.1)

        if self.offloading_decision:
            # Do not use this offloading decision because we are already done
            # Just unlock the other waiting node and continue
            self.logger.info(f'{self.id} is not offloading to {self.offloading_decision["node-id"]}; Reason: training already done')
            self.logger.info(f'Offloading request to be ignored -> Unlocking client {self.offloading_decision["node-id"]}')
            self.message_async(self.offloading_decision['node-id'], Client.unlock)
            self.offloading_decision = {}

        if self.has_offloading_request:
            offloading_train_start = time.time()
            offloading_response_id = self.offloading_reponse_id
            self.logger.info(f'Available keys in nets dict: {self.nets.keys()}')
            self.logger.info(f'Other keys in nets dict: {self.nets.other_keys()}')
            other_client_id = self.nets.other_keys()[0]
            self.logger.info(f'I need to train the offloading model from client {other_client_id} as well!')
            self.nets.select(other_client_id)
            # self.logger.info(self.nets)
            # loss, weights = self.train(round_id, num_epochs, False, 0)
            loss, weights, num_samples = self.train(round_id, num_epochs, False, self.offloading_rem_local_updates)
            offloading_time_mark_between = time.time()
            # time_mark_between = time.time()
            accuracy, test_loss = self.test()
            offloading_test_end = time.time()
            offloading_train_duration= offloading_time_mark_between - offloading_train_start
            offloading_test_duration = offloading_test_end - offloading_time_mark_between
            self.logger.info('Stopping training of alternative model')
            for k, v in weights.items():
                weights[k] = v.cpu()
            self.message_async(server_ref, 'receive_training_result', offloading_response_id,
                                   [loss, weights, accuracy, test_loss, round_duration, offloading_train_duration, offloading_test_duration, num_samples])
            self.nets.reset()
            # @TODO: Use offloading response_id to send offloaded weights to the server
            trained_offloaded_model = self.nets.remove_model(other_client_id)
            self.has_offloading_request = False


        # if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
        #     self.optimizer.pre_communicate()
        # for k, v in weights.items():
        #     weights[k] = v.cpu()
        self.logger.info('Ending training')
        # return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')