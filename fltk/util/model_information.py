from typing import Tuple
from fltk.datasets.loader_util import get_dataset
from fltk.util.definitions import Dataset, LogLevel
from dataclasses import dataclass
from fltk.util.config import Config
import pandas as pd


@dataclass
class DataSetInformation:
    name: str
    input_size: Tuple[int]
    num_train: int
    num_test: int
    batch_size: int  # Batch size used when retrieving the input shape
    random_input: any
    num_classes: int = 0


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
    print(datasets_df.drop('Random Input', axis=1).to_markdown(tablefmt="grid"))
    print(
        'Note: The batch size in the table is the size used to retrieve the shape of the input data. '
        'This can be changed in the config')


if __name__ == '__main__':
    main()
