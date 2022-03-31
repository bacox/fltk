import time
from typing import Tuple, Any

import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import Config


class Client(Node):
    running = False
    terminate_training = False
    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        super().__init__(id, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
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

    def stop_client(self):
        self.logger.info('Got call to stop event loop')
        self.running = False

    def _event_loop(self):
        self.logger.info('Starting event loop')
        while self.running:
            time.sleep(0.1)
        self.logger.info('Exiting node')

    def train(self, num_epochs: int):
        start_time = time.time()

        running_loss = 0.0
        final_running_loss = 0.0
        if self.distributed:
            self.dataset.train_sampler.set_epoch(num_epochs)

        number_of_training_samples = len(self.dataset.get_train_loader())
        # self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')

        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            # Mark logging update step
            if i % self.config.log_interval == 0:
                self.logger.info(
                    '[%s] [%d, %5d] loss: %.3f' % (self.id, num_epochs, i, running_loss / self.config.log_interval))
                final_running_loss = running_loss / self.config.log_interval
                running_loss = 0.0

            if self.terminate_training:
                break

        end_time = time.time()
        duration = end_time - start_time
        # self.logger.info(f'Train duration is {duration} seconds')

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
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                if self.terminate_training:
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

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

        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, v in weights.items():
            weights[k] = v.cpu()
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')