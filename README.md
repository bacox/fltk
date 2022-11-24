# FLTK - Federation Learning Toolkit
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

This toolkit is can be used to run Federated Learning experiments.
Pytorch Distributed ([docs](https://pytorch.org/tutorials/beginner/dist_overview.html)) is used in this project.
The goal if this project is to launch Federated Learning nodes in truly distribution fashion.

This project is tested with Ubuntu 20.04 and python {3.7, 3.8}.
### Global idea
Pytorch distributed works based on a world_size and ranks. The ranks should be between 0 and world_size-1.
Generally, the federator has rank 0 and the clients have ranks between 1 and world_size-1.

General protocol:

1. Client selection by the federator
2. The selected clients download the model.
2. Local training on the clients for X number of epochs
3. Weights/gradients of the trained model are send to the federator
4. Federator aggregates the weights/gradients to create a new and improved model
5. Updated model is shared to the clients
6. Repeat step 1 to 5 until convergence

Important notes:

* Data between clients is not shared to each other
* The data is non-IID
* Hardware can heterogeneous
* The location of devices matters (network latency and bandwidth)
* Communication can be costly

## Project structure
Structure with important folders and files explained:
```
project
├── experiments
├── deploy                                    # Templates for automatic deployment  
│     └── docker                              # Describes how a systems is deployed using docker containers
│          ├── stub_default.yml
│          └── system_stub.yml                # Describes the federator and the network
├── fltk                                      # Source code
│     ├── core                                # Different dataset definitions
│     ├── datasets                            # Different dataset definitions
│     ├── nets                                # Available networks
│     ├── samplers                            # Different types of data samplers to create non-IID distributions
│     ├── schedulers                          # Learning Rate Schedulers
│     ├── strategy                            # Client selection and model aggregation algorithms
│     ├── util                                # Various utility functions
│     ├── datasets                            # Different dataset definitions
│     │    ├── data_distribution              # Datasets with distributed sampler
│     │    └── distributed                    # "regular" datasets for centralized use
│     └── __main__.py                         # Main package entrypoint
├── Dockerfile                                # Dockerfile to run in containers
├── LICENSE
├── README.md
└── setup.py
```

## Execution modes
Federatd Learning experiments can be set up in various ways (Simulation, Emulation, or fully distributed). Not all have the same requirements and thus some setup are more suited then others depending on the experiment.

### Simulation
With the method as single machine is used to execute all the different nodes in the system.
The execution is done in a sequential manner, i.e. first node 1 is executed, then node 2, and so on. One of the upsides of this option is the ability to use GPU acceleration for the computations.

### Docker-Compose (Emulation)
With systems like docker we can emulate a federated learning system on a single machine. Each node is allocated to one or more CPU cores and executed in an isolated container. This allows for real-time experiments where timing is important and where the execution of clients have effect on eachother. Docker also allows for containers to be limited by CPU speed, RAM, and network properties.

### Real distributed (Google Cloud)
In this case, the code is deployed natively on a machine, for example a cluster. 
This is the closest real-world approximation when experimenting with Federated Learning systems. This allows for real-time experiments where timing is important and where the execution of clients have effect on eachother. A downside of this method is the shear number of machines needed to run an experiment. Additionally the compute speed and other hardware spcifications are more difficult to limit.

### Hybrid
The Docker (Compose) and real-distributed method can be mixed in a hybrid system. For example two servers can run a set of docker containers that are linked to each other. Similarly, a set of docker images on a server can participate in a system with real distributed machines. 

## Models

* Cifar10-CNN
* Cifar10-ResNet
* Cifar100-ResNet
* Cifar100-VGG
* Fashion-MNIST-CNN
* Fashion-MNIST-ResNet
* Reddit-LSTM
* Shakespeare-LSTM

## Datasets

* Cifar10
* Cifar100
* Fashion-MNIST
* MNIST
* Shakespeare

## Prerequisites

When running in docker containers the following dependencies need to be installed:

* Docker
* Docker-compose

## Install
```bash
python3 -m pip install -r requirements.txt
```

### Load models
```bash
python3 -m fltk.util.default_models
```

## Examples
<details><summary>Show Examples</summary>

<p>


### Docker compose
**Note:** Make sure docker and docker-compose are installed.

Generate docker configuration
```bash
python3 -m fltk util-generate experiments/example_docker/
```
Run example experiment
```bash
python3 -m fltk util-run experiments/example_docker/
```

### Single machine (Native)

#### Launch single client
Launch Federator
```bash
python3 -m fltk single configs/experiment.yaml --rank=0
```
Launch Client
```bash
python3 -m fltk single configs/experiment.yaml --rank=1
```

#### Spawn FL system
```bash
python3 -m fltk spawn configs/experiment.yaml
```

### Two machines (Native)
To start a cross-machine FL system you have to configure the network interface connected to your network.
For example, if your machine is connected to the network via the wifi interface (for example with the name `wlo1`) this has to be configured as shown below:
```bash
os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'
os.environ['TP_SOCKET_IFNAME'] = 'wlo1'
```
Use `ifconfig` to find the name of the interface name on your machine.


### Google Cloud Platform
See Manual on brightspace

</p>
</details>

## Known issues

* Currently, there is no GPU support docker containers (or docker compose)
* First epoch only can be slow (6x - 8x slower)