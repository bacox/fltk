######### Definitions #########
# 1. Datasets                 #
# 2. Networks (models)        #
# 3. Aggregation methods      #
# 4. Client selection methods #
# 5. Data samplers            #
# 6. Optimizers               #
###############################
# Use enums instead of dataclasses?
from enum import Enum, unique


@unique
class DataSampler(Enum):
    uniform = "uniform"
    q_sampler = "q sampler"
    limit_labels = "limit labels"
    dirichlet = "dirichlet"
    limit_labels_q = "limit labels q"
    emd_sampler = 'emd sampler'
    limit_labels_flex = "limit labels flex"
    n_labels = "n labels"


@unique
class Optimizations(Enum):
    sgd = 'SGD'
    fedprox = 'FedProx'
    fednova = 'FedNova'


@unique
class Dataset(Enum):
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    fashion_mnist = 'fashion-mnist'
    mnist = 'mnist'
    shakespeare = 'shakespeare'


class LogLevel(Enum):
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


@unique
class Aggregations(Enum):
    avg = 'Avg'
    fedavg = 'FedAvg'
    sum = 'Sum'


@unique
class Nets(Enum):
    cifar100_resnet = "Cifar100ResNet"
    cifar100_vgg = "Cifar100VGG"
    cifar10_cnn = "Cifar10CNN"
    cifar10_resnet = "Cifar10ResNet"
    fashion_mnist_cnn = "FashionMNISTCNN"
    fashion_mnist_resnet = "FashionMNISTResNet"
    mnist_cnn = 'MNISTCNN'
    shakespeare_rnn = 'ShakespeareRNN'


@unique
class Algorithm(Enum):
    vanilla = 'vanilla'
    tifl = 'tifl'
    deadline = 'deadline'
    offloading = 'offloading'
