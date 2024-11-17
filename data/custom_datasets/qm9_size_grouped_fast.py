import os
import torch
import h5py
import pickle
import random
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, Batch, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import Sampler
from torch_geometric.data import download_url
from functools import lru_cache


class SizeGroupedQM9(InMemoryDataset):
    """
    QM9 dataset that ensures graphs in batches have the same size and maintains
    one-to-one correspondence between PyG graphs and HDF5 data.
    """
    url = {
        'test': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAxhVewejSl7gVMACa1tBUda/QM9_test.p?dl=1',
        'valid': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAOfEx-jGC6vvi43fh0tOq6a/QM9_val.p?dl=1',
        'train': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AADtx0EMRz5fhUNXaHFipkrza/QM9_train.p?dl=1',
    }

    def __init__(self, root, split='train', transform=None, pre_transform=None, force_process=False, debug=False):
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.force_process = force_process
        self.debug = debug

        super().__init__(root, transform, pre_transform)

        # Check for processed data if force_process is False
        if not self.force_process and self.processed_files_exist():
            # Load processed data
            self.load_processed_data()
        else:
            # Process the data if it doesn’t exist or if force_process is True
            self.process()

        # Group indices by node count
        self.size_groups = defaultdict(list)
        for idx, num_nodes in enumerate(self.node_counts):
            self.size_groups[num_nodes].append(idx)

    @property
    def raw_file_names(self):
        return ['QM9_test.p', 'QM9_val.p', 'QM9_train.p']

    def processed_file_names(self):
        return [
            f'{self.split}_data.pt',
            f'{self.split}_slices.pt',
            f'{self.split}_node_counts.pt',
            f'{self.split}_distance_mats.pt'
        ]

    def download(self):
        for split, url in self.url.items():
            filename = f'QM9_{split}.p' if split != 'valid' else 'QM9_val.p'
            download_url(url=url, folder=self.raw_dir, filename=filename)

    def processed_files_exist(self):
        """Check if processed files already exist for the split."""
        return all(os.path.exists(os.path.join(self.processed_dir, file)) for file in self.processed_file_names())

    def process(self):
        """Process raw data into PyG graphs and save matrix data as PyTorch tensors."""
        print(f"Processing {self.split} split...")

        # Load raw data
        raw_file = f'QM9_{self.split if self.split != "valid" else "val"}.p'
        with open(os.path.join(self.raw_dir, raw_file), 'rb') as f:
            raw_graphs = pickle.load(f)

        # If debug mode, limit to 16 graphs
        if self.debug:
            print("Debug mode is ON: Processing only 16 graphs.")
            raw_graphs = raw_graphs[:16]

        # Process graphs and collect metadata
        pyg_graphs = []
        node_counts = []
        distance_mats = []
        affinities = []
        edge_features_list = []

        # Process each graph
        for idx, raw_graph in enumerate(tqdm(raw_graphs)):
            try:
                # Convert to PyG format
                pyg_graph = self._convert_to_pyg(raw_graph)

                # Apply pre_transform if defined
                if self.pre_transform is not None:
                    pyg_graph = self.pre_transform(pyg_graph)

                num_nodes = pyg_graph.num_nodes

                # Collect matrix data
                usable_features = raw_graph['usable_features']
                if 'distance_mat' in usable_features:
                    distance_mats.append(torch.tensor(usable_features['distance_mat'], dtype=torch.float))
                else:
                    distance_mats.append(None)  # Handle missing data if necessary

                if 'affinity' in usable_features:
                    affinities.append(torch.tensor(usable_features['affinity'], dtype=torch.float))
                else:
                    affinities.append(None)

                if 'edge_features' in usable_features:
                    edge_features_list.append(torch.tensor(usable_features['edge_features'], dtype=torch.float))
                else:
                    edge_features_list.append(None)

                # Collect PyG graph and metadata
                pyg_graphs.append(pyg_graph)
                node_counts.append(num_nodes)

            except Exception as e:
                print(f"Error processing graph {idx}: {str(e)}")
                continue

        # Save processed data
        data, slices = self.collate(pyg_graphs)
        torch.save(data, os.path.join(self.processed_dir, f'{self.split}_data.pt'))
        torch.save(slices, os.path.join(self.processed_dir, f'{self.split}_slices.pt'))
        torch.save(node_counts, os.path.join(self.processed_dir, f'{self.split}_node_counts.pt'))

        # Save matrix data as PyTorch tensors
        torch.save(distance_mats, os.path.join(self.processed_dir, f'{self.split}_distance_mats.pt'))
        torch.save(affinities, os.path.join(self.processed_dir, f'{self.split}_affinities.pt'))
        torch.save(edge_features_list, os.path.join(self.processed_dir, f'{self.split}_edge_features.pt'))


    def _convert_to_pyg(self, raw_graph):
        """Convert raw graph data to PyG format."""
        x = torch.tensor(raw_graph['usable_features']['x'], dtype=torch.float)
        edge_index = torch.tensor(raw_graph['original_features']['edge_index'], dtype=torch.long)
        pos = torch.tensor(raw_graph['original_features']['pos'], dtype=torch.float)
        edge_attr = torch.tensor(raw_graph['original_features']['edge_attr'], dtype=torch.float)
        y = torch.tensor(raw_graph['y'], dtype=torch.float)

        # Validate data
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        assert edge_index.max() < x.size(0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)

    def load_processed_data(self):
        """Load processed data and node counts."""
        self.data, self.slices = self.load_data_slices()
        node_counts_path = os.path.join(self.processed_dir, f'{self.split}_node_counts.pt')
        self.node_counts = torch.load(node_counts_path)

        # Load matrix data
        self.distance_mats = torch.load(os.path.join(self.processed_dir, f'{self.split}_distance_mats.pt'))
        self.affinities = torch.load(os.path.join(self.processed_dir, f'{self.split}_affinities.pt'))
        self.edge_features_list = torch.load(os.path.join(self.processed_dir, f'{self.split}_edge_features.pt'))


    def load_data_slices(self):
        """Load PyG graph data and slices."""
        data = torch.load(os.path.join(self.processed_dir, f'{self.split}_data.pt'))
        slices = torch.load(os.path.join(self.processed_dir, f'{self.split}_slices.pt'))
        return data, slices

    def len(self):
        return len(self.node_counts)

    def get(self, idx):
        """Get a graph and its corresponding matrix data."""
        # Get PyG graph
        data = self._get_graph_data(idx)

        # Add matrix data
        if self.distance_mats[idx] is not None:
            data.distance_mat = self.distance_mats[idx]
        if self.affinities[idx] is not None:
            data.affinity = self.affinities[idx]
        if self.edge_features_list[idx] is not None:
            data.edge_features = self.edge_features_list[idx]

        return data

    def _get_graph_data(self, idx):
        """Get PyG graph data for given index."""
        data = Data()
        for key in self.data.keys():

            if key in ['num_nodes']:
                continue

            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()

            # Handle 2D tensors (like edge_index) with slicing across both dimensions
            if key in ['edge_index']:
                data[key] = item[:, start:end]
            else:
                data[key] = item[start:end]

        return data


class SameSizeBatchSampler(Sampler):
    """Batch sampler that groups graphs of the same size together."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _create_batches(self):
        """Create batches of same-size graphs, including partial batches if needed."""
        batches = []

        # Process each size group
        for size, indices in self.dataset.size_groups.items():
            # Copy and shuffle indices within each size group
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)

            # Split indices into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                # Add all batches, even if they are smaller than batch_size
                if len(batch) > 1:  # <--- keeping it to 1, for legal batch normalization
                    batches.append(batch)

        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_with_matrices(batch):
    """Collate function that handles both PyG graphs and matrix data."""
    batch_data = Batch.from_data_list(batch)
    return batch_data


def create_same_size_dataloader(dataset, batch_size, shuffle, num_workers=0):
    """Create a dataloader that returns batches of same-size graphs."""
    batch_sampler = SameSizeBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_with_matrices,
        num_workers=num_workers
    )
