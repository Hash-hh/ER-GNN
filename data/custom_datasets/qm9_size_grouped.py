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

    def __init__(self, root, split='train', transform=None, pre_transform=None, force_process=False):
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.force_process = force_process

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
            'distance_affinity.h5'
        ]

    def download(self):
        for split, url in self.url.items():
            filename = f'QM9_{split}.p' if split != 'valid' else 'QM9_val.p'
            download_url(url=url, folder=self.raw_dir, filename=filename)

    def processed_files_exist(self):
        """Check if processed files already exist for the split."""
        return all(os.path.exists(os.path.join(self.processed_dir, file)) for file in self.processed_file_names())

    def process(self):
        """Process raw data into PyG graphs and HDF5 matrices."""
        print(f"Processing {self.split} split...")

        # Load raw data
        raw_file = f'QM9_{self.split if self.split != "valid" else "val"}.p'
        with open(os.path.join(self.raw_dir, raw_file), 'rb') as f:
            raw_graphs = pickle.load(f)

        # Process graphs and collect metadata
        pyg_graphs = []
        node_counts = []

        # Create HDF5 file for matrix data
        h5f_path = os.path.join(self.processed_dir, 'distance_affinity.h5')
        with h5py.File(h5f_path, 'a') as h5f:
            # Create or get split group
            if self.split in h5f:
                del h5f[self.split]
            split_group = h5f.create_group(self.split)

            # Process each graph
            for idx, raw_graph in enumerate(tqdm(raw_graphs)):
                try:
                    # Convert to PyG format
                    pyg_graph = self._convert_to_pyg(raw_graph)
                    num_nodes = pyg_graph.num_nodes

                    # Store matrix data in HDF5
                    graph_group = split_group.create_group(f'graph_{idx}')
                    graph_group.attrs['num_nodes'] = num_nodes

                    # Store distance matrix, affinity, and edge features
                    for key in ['distance_mat', 'affinity', 'edge_features']:
                        if key in raw_graph['usable_features']:
                            data = raw_graph['usable_features'][key]
                            if key == 'distance_mat':
                                assert data.shape == (num_nodes, num_nodes)
                            graph_group.create_dataset(key, data=data)

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
        self.h5f_path = os.path.join(self.processed_dir, 'distance_affinity.h5')

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

        # Add matrix data from HDF5
        with h5py.File(self.h5f_path, 'r') as h5f:
            graph_group = h5f[f'{self.split}/graph_{idx}']

            # Load and validate matrix data
            for key in ['distance_mat', 'affinity', 'edge_features']:
                if key in graph_group:
                    setattr(data, key, torch.tensor(graph_group[key][:]))

        return data

    # def get(self, idx):
    #     """Get a graph and its corresponding matrix data with lazy loading."""
    #     # Get PyG graph
    #     data = self._get_graph_data(idx)
    #
    #     # Attach lazy-loaded HDF5 data access methods
    #     data.distance_mat = self._lazy_load_hdf5(idx, 'distance_mat')
    #     data.affinity = self._lazy_load_hdf5(idx, 'affinity')
    #     data.edge_features = self._lazy_load_hdf5(idx, 'edge_features')
    #
    #     return data
    #
    # @lru_cache(maxsize=1024)
    # def _lazy_load_hdf5(self, idx, key):
    #     """Lazy load matrix data for a given graph and attribute key."""
    #     with h5py.File(self.h5f_path, 'r') as h5f:
    #         graph_group = h5f[f'{self.split}/graph_{idx}']
    #         if key in graph_group:
    #             return torch.tensor(graph_group[key][:])
    #     return None  # Return None if key is not found

    def _get_graph_data(self, idx):
        """Get PyG graph data for given index."""
        data = Data()
        for key in self.data.keys():
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
    graphs = []
    distance_mats = []
    affinities = []
    edge_features_list = []

    for data in batch:
        # Create new Data object for PyG graph
        graphs.append(Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            pos=data.pos,
            y=data.y
        ))

        # Collect matrix data
        distance_mats.append(data.distance_mat)
        affinities.append(data.affinity)
        edge_features_list.append(data.edge_features)

    # Create final batch
    batch_data = Batch.from_data_list(graphs)
    batch_data.distance_mats = torch.stack(distance_mats)
    batch_data.affinities = torch.stack(affinities)
    batch_data.edge_features = torch.stack(edge_features_list)

    return batch_data


def create_same_size_dataloader(dataset, batch_size, shuffle, num_workers=0):
    """Create a dataloader that returns batches of same-size graphs."""
    # dataset = SizeGroupedQM9(root=root, split=split)
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