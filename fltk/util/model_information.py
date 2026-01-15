import os
from typing import Tuple

import torch
from torchsummary import summary

from fltk.datasets.loader_util import get_dataset
from fltk.util.definitions import Dataset, LogLevel, Nets
from dataclasses import dataclass
from fltk.util.config import Config
import pandas as pd
from fltk.nets import get_net_by_name

model_dataset_match = {
    Nets.mnist_cnn: Dataset.mnist,
    Nets.cifar10_cnn: Dataset.cifar10,
    Nets.cifar10_resnet: Dataset.cifar10,
    Nets.cifar100_vgg: Dataset.cifar100,
    Nets.cifar100_resnet: Dataset.cifar100,
    Nets.fashion_mnist_cnn: Dataset.fashion_mnist,
    Nets.fashion_mnist_resnet: Dataset.fashion_mnist
}


@dataclass
class DataSetInformation:
    name: str
    input_size: Tuple[int]
    num_train: int
    num_test: int
    batch_size: int  # Batch size used when retrieving the input shape
    random_input: any
    num_classes: int = 0


def print_header(text: str, padding: int = 2, fill: str = '*'):
    length = len(text) + 2 * padding + 2
    print(fill * length)
    print(f'{fill}{" " * padding}{text}{" " * padding}{fill}')
    print(fill * length)


def get_dataset_information(d: Dataset, cfg: Config) -> list:
    ds = get_dataset(d)(cfg)
    num_train = len(ds.get_train_loader())
    num_test = len(ds.get_test_loader())
    num_classes = 0
    if hasattr(ds.train_dataset, 'classes'):
        num_classes = len(ds.train_dataset.classes)
    input_data, _labels = next(iter(ds.get_train_loader()))
    input_size = input_data.size()
    return [str(d), tuple(input_size), num_train, num_test, cfg.batch_size, input_data, num_classes]


def load_model_from_file(model_class, config: Config):
    """
    Load a model from a file.
    """
    # model_class = get_net_by_name(config.net_name)
    model = model_class()

    model_file_path = os.path.join(config.get_default_model_folder_path(), model_class.__name__ + ".model")

    if os.path.exists(model_file_path):
        try:
            model.load_state_dict(torch.load(model_file_path))
        except:
            print("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

            model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    else:
        print("Could not find model: {}".format(model_file_path))
    return model


def main():
    """
    Creates an overview of the available Datasets and their basic information.
    The output is written to output as a Markdown table.
    Note: Currently, the Shakespeare dataset is ignored in this script.
    """
    cfg = Config()
    cfg.rank = 0
    cfg.world_size = 2
    cfg.batch_size = 1
    cfg.log_level = LogLevel.WARN
    datasets_info = [get_dataset_information(d, cfg) for d in Dataset.__iter__() if d != Dataset.shakespeare]
    datasets_df = pd.DataFrame(datasets_info,
                               columns=['Name', 'Input Size', 'Train samples', 'Test Samples', 'Batch Size',
                                        'Random Input', 'Num Classes'])
    print_header('Model information')
    for net_name, ds_name in model_dataset_match.items():
        if net_name == Nets.fashion_mnist_resnet:
            break
        print(net_name, ds_name)
        model_class = get_net_by_name(net_name)
        model = load_model_from_file(model_class, cfg)
        input_point = datasets_df[datasets_df['Name'] == str(ds_name)]['Random Input'].values[0]
        input_size = tuple(input_point.size())[1:]
        print(input_size)
        summary(model, input_size, batch_size=cfg.batch_size, device='cpu')
        print('')
    print_header('Dataset information')
    print(datasets_df.drop('Random Input', axis=1).to_markdown(tablefmt="grid"))
    print(
        'Note: The batch size in the table is the size used to retrieve the shape of the input data. '
        'This can be changed in the config')


if __name__ == '__main__':
    main()
