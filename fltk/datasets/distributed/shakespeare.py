from __future__ import annotations

from datasets.leaf.pickle_dataset import PickleDataset
from fltk.datasets import DistDataset
from torch.utils.data import DataLoader
from fltk.samplers import get_sampler
from util.config import Config


class DistShakespeareDataset(DistDataset):

    def __init__(self, args: Config):
        super(DistShakespeareDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
    
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Shakespeare train data")

        # get vocab and index data
        dataset = 'shakespeare'
        data_root: str = None
        pickle_root: str = None
        pdataset = PickleDataset(dataset_name=dataset, data_root=data_root, pickle_root=pickle_root)
        # client_id = None
        client_id = self.args.rank
        self.train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id=client_id)
        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Shakespeare test data")
        dataset = 'shakespeare'
        data_root: str = None
        pickle_root: str = None
        pdataset = PickleDataset(dataset_name=dataset, data_root=data_root, pickle_root=pickle_root)
        # client_id = None
        client_id = self.args.rank
        self.test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id=client_id)
        self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, sampler=self.test_sampler)