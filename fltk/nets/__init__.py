from .cifar_10_cnn import Cifar10CNN
from .cifar_100_resnet import Cifar100ResNet
from .fashion_mnist_cnn import FashionMNISTCNN
from .fashion_mnist_resnet import FashionMNISTResNet
from .cifar_10_resnet import Cifar10ResNet
from .cifar_100_vgg import Cifar100VGG
from .mnist_cnn import MNIST_CNN
from .shakespeare_rnn import RNN_Shakespeare
from fltk.util.definitions import Nets


def available_nets():
    return {
        Nets.cifar100_resnet: Cifar100ResNet,
        Nets.cifar100_vgg: Cifar100VGG,
        Nets.cifar10_cnn: Cifar10CNN,
        Nets.cifar10_resnet: Cifar10ResNet,
        Nets.fashion_mnist_cnn: FashionMNISTCNN,
        Nets.fashion_mnist_resnet: FashionMNISTResNet,
        Nets.mnist_cnn: MNIST_CNN,
        Nets.shakespeare_rnn: RNN_Shakespeare

    }

def get_net_by_name(name: Nets):
    return available_nets()[name]


def get_net_split_point(name: Nets):
    nets_split_point = {
        Nets.cifar100_resnet: 48,
        Nets.cifar100_vgg: 28,
        Nets.cifar10_cnn: 15,
        Nets.cifar10_resnet: 39,
        Nets.fashion_mnist_cnn: 7,
        Nets.fashion_mnist_resnet: 7,
        Nets.mnist_cnn: 2,
        Nets.shakespeare_rnn: 2
    }
    return nets_split_point[name]

def get_net_feature_layers_names(name: Nets):
    nets_split_point = {
        Nets.cifar100_resnet: [],
        Nets.cifar100_vgg: [],
        Nets.cifar10_cnn: ['conv1','bn1', 'conv2','bn2', 'conv3','bn3', 'conv4','bn4', 'conv5','bn5', 'conv6','bn6',],
        Nets.cifar10_resnet: [],
        Nets.fashion_mnist_cnn: ['layer1.0', 'layer1.1', 'layer1.1', 'layer2.0', 'layer2.1'],
        Nets.fashion_mnist_resnet: [],
        Nets.mnist_cnn: ['conv1', 'conv2'],
        Nets.shakespeare_rnn: []
    }
    return nets_split_point[name]