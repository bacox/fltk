from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from fltk.datasets.distributed.dataset import DistDataset
from fltk.samplers import get_sampler


class DistCIFAR100Dataset(DistDataset):

    def __init__(self, args):
        super(DistCIFAR100Dataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' CIFAR100 train data")
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        self.train_dataset = datasets.CIFAR100(root=self.get_args().get_data_path(), train=True, download=True,
                                              transform=transform)
        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' CIFAR100 test data")

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.test_dataset = datasets.CIFAR100(root=self.get_args().get_data_path(), train=False, download=True,
                                        transform=transform)
        self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, sampler=self.test_sampler)


    def load_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' CIFAR100 train data")

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.CIFAR100(root=self.get_args().get_data_path(), train=True, download=True,
                                         transform=transform)
        sampler = get_sampler(self.test_dataset, self.args)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        train_data = self.get_tuple_from_data_loader(train_loader)
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Finished loading '{dist_loader_text}' CIFAR100 train data")

        return train_data

    def load_test_dataset(self):
        self.logger.debug("Loading CIFAR100 test data")

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR100(root=self.get_args().get_data_path(), train=False, download=True,
                                        transform=transform)
        sampler = get_sampler(self.test_dataset, self.args)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.logger.debug("Finished loading CIFAR10 test data")

        return test_data

