import copy
import time
from itertools import combinations, product

from tqdm import tqdm

from fltk.core.node import Node
from fltk.datasets.loader_util import get_dataset
from fltk.util.definitions import Dataset, Nets, LogLevel
import numpy as np
import torch
from fltk.nets import get_net_split_point
from fltk.util.profilerV3 import Profiler
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import Config
from fltk.util.log import getLogger


def get_dataloader(dataset: Dataset, net: Nets):
    pass


class TestClient(Node):
    def init_dataloader(self, world_size: int = None):
        config = copy.deepcopy(self.config)
        if world_size:
            config.world_size = world_size
        self.logger.info(f'world size = {config.world_size} with rank={config.rank}')
        self.dataset = get_dataset(config.dataset_name)(config)
        self.labels_dist = []
        # for _, label in self.dataset.get_train_loader():
        #     self.labels_dist.append(label)
        # # self.labels_dist = torch.unique(torch.stack(self.labels_dist), return_counts=True)
        # self.labels_dist = torch.unique(torch.cat(self.labels_dist, dim=0), return_counts=True)
        self.finished_init = True
        self.logger.info('Done with init')

    # def logger(self, msg: str):
    #     print(msg)

    def __init__(self, config, id: int, rank: int, world_size: int):
        super().__init__(id, rank, world_size, config)
        self.offloading_decision = {}
        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.nets.selected().parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.logger, self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

        self.init_dataloader(world_size=2)


    def run(self, use_profiler: bool, profiling_size: int = 0, num_epochs: int = 1):
        device = torch.device("cpu")
        net_split_point = get_net_split_point(self.config.net_name)
        number_of_training_samples = len(self.dataset.get_train_loader()) * num_epochs
        remaining_training_samples = number_of_training_samples
        profiling_size = min(profiling_size, number_of_training_samples)
        if not profiling_size:
            profiling_size = number_of_training_samples
        p_data = np.zeros(profiling_size)
        iter = 0
        p = Profiler(profiling_size, net_split_point - 1, use_profiler)
        network = self.nets.selected()
        running_loss = 0
        test_start_t = time.time()
        if p.active:
            p.attach(network)
        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            s_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

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
                    '[%s] [%d, %5d] loss: %.3f' % (self.id, 0, i, running_loss / self.config.log_interval))
            #     final_running_loss = running_loss / self.config.log_interval
            #     running_loss = 0.0

            # if limit_num_training_updates and i >= limit_num_training_updates:
            #     break

            if p.active:
                if i == profiling_size - 1:
                    p.active = False
                    p.remove_all_handles()
                    profiler_data = p.aggregate_values()
                    # self.logger.info(f'Profiler data: {profiler_data}')
                    # self.logger.info(f'{remaining_training_samples=}')
                    #
                    # self.logger.info(f'Profiler data (sum): {np.sum(profiler_data)} and {e_time} and {np.mean(p_data)}')
                    # self.logger.info(
                    #     f'Profiler data (%): {np.abs(np.mean(p_data) - np.sum(profiler_data)) / np.sum(profiler_data)}')
                    profiling_obj = {
                        'pm': profiler_data,
                        'ps': profiling_size,
                        'bs': self.config.batch_size,
                        'rl': number_of_training_samples - i,
                        # 'dd': dict(zip(self.labels_dist[0].tolist(), self.labels_dist[1].tolist()))
                    }
                    # self.message_async('federator', 'save_performance_metric', self.id, profiling_obj)
        test_stop_t = time.time()

        return test_stop_t - test_start_t, use_profiler

def get_variants():

    batch_sizes = [16, 32, 64]
    profiling_size = [0, 100]
    # num_epochs = [1,2,4,8]
    num_epochs = [1]
    repetition_id = list(range(1))
    net_dicts = [
        [Nets.mnist_cnn, Dataset.mnist],
        [Nets.fashion_mnist_cnn, Dataset.fashion_mnist],
        [Nets.fashion_mnist_resnet, Dataset.fashion_mnist],
        # [Nets.cifar10_cnn, Dataset.cifar10],
        # [Nets.cifar10_resnet, Dataset.cifar10],
        # [Nets.cifar100_vgg, Dataset.cifar100],
        # [Nets.cifar100_resnet, Dataset.cifar100],
    ]

    return list(product(net_dicts, batch_sizes, profiling_size, num_epochs, repetition_id))


if __name__ == '__main__':
    # print('Starting test')
    # variants = get_variants()
    collected_data = []
    variants = get_variants()
    idx = 1
    total = len(variants)
    for [network, dataset], batch_size, profiling_size, num_epoch, repetition_id in tqdm(variants):
        # print(f'Starting run [{idx}/{total}]')
        # print(f'Using the following values: n: {network}, d:{dataset}, b:{batch_size}, p:{profiling_size}, num:{num_epoch}, r:{repetition_id}')

        # l = getLogger(__name__, LogLevel.WARN)
        config = Config()
        config.log_level = LogLevel.ERROR
        config.net_name = network
        config.rank = 1
        config.dataset_name = dataset
        config.batch_size = batch_size
        world_size = 5
        config.world_size = world_size
        config.distributed = True
        #  config, id: int, rank: int, world_size: int
        c = TestClient(config, 0, 1, world_size)
        # print('Starting run without profiler')
        time_no_p, _ = c.run(False, profiling_size=profiling_size, num_epochs=num_epoch)
        # print('Starting run with profiler')
        time_with_p, _ = c.run(True, profiling_size=profiling_size, num_epochs=num_epoch)

        # print(f'Time with profiler is {time_with_p} seconds')
        # print(f'Time no profiler is {time_no_p} seconds')

        # print(f'Result of : n: {network}, d:{dataset}, b:{batch_size}, p:{profiling_size}, num:{num_epoch}, r:{repetition_id}')

        collected_data.append([time_with_p, True, network, dataset, batch_size, profiling_size, num_epoch, repetition_id])
        collected_data.append([time_no_p, False, network, dataset, batch_size, profiling_size, num_epoch, repetition_id])
        idx += 1
        # dl = get_dataloader(d, n)
        # break

    print('Results')
    [print('='*15) for _ in range(3)]
    for time_p, use_profiler, network, dataset, batch_size, profiling_size, num_epoch, repetition_id in collected_data:
        # print(f'Using the following values: n: {network}, d:{dataset}, b:{batch_size}, p:{profiling_size}, num:{num_epoch}, r:{repetition_id}')
        print(f'Time is {time_p} seconds with values -> profiler:{use_profiler}, n: {network}, d:{dataset}, b:{batch_size}, p:{profiling_size}, num:{num_epoch}, r:{repetition_id}')
