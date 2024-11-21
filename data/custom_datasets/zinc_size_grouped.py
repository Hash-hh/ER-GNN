import os
import os.path as osp
from collections import defaultdict
from typing import Callable, Optional, List
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import pickle
from torch_geometric.io import fs
from torch_geometric.datasets import ZINC


class SizeGroupedZINC(InMemoryDataset):

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')


    def __init__(self,
                 root: str,
                 subset: bool = False,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_process: bool = False):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}; must be 'train', 'val', or 'test'."
        self.split = split
        self.subset = subset
        self.force_process = force_process

        super().__init__(root, transform, pre_transform)

        if not self.force_process and self._processed_files_exist():
            self._load_processed_data()
        else:
            self.process()

        # Group indices by node count
        self.size_groups = defaultdict(list)
        for idx, num_nodes in enumerate(self.node_counts):
            self.size_groups[num_nodes].append(idx)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.pickle', 'val.pickle', 'test.pickle', 'train.index', 'val.index', 'test.index']

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f'{self.split}_data.pt',
            f'{self.split}_slices.pt',
            f'{self.split}_node_counts.pt',
            f'{self.split}_dense_edge_features.pt'
        ]

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        print(f"Processing {self.split} split...")

        raw_file = osp.join(self.raw_dir, f'{self.split}.pickle')
        with open(raw_file, 'rb') as f:
            mols = pickle.load(f)

        indices = list(range(len(mols)))

        if self.subset:
            subset_index_file = osp.join(self.raw_dir, f'{self.split}.index')
            with open(subset_index_file) as f:
                indices = [int(x) for x in f.read().strip().split(',')]

        data_list = []
        node_counts = []
        dense_edge_features_list = []
        # dense_edge_features_onehot_list = []
        max_edge_type = 4  # Assuming edge types are in the range [0, 3]

        pbar = tqdm(total=len(indices), desc=f'Processing {self.split} dataset')

        for idx in indices:
            mol = mols[idx]

            # Extract node features and labels
            x = mol['atom_type'].to(torch.long).view(-1, 1)  # Shape: [num_nodes, 1]
            y = mol['logP_SA_cycle_normalized'].to(torch.float)  # Shape: [1]

            # Extract adjacency matrix and edge features
            adj = mol['bond_type']  # Shape: [num_nodes, num_nodes]
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # Shape: [2, num_edges]
            edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)  # Shape: [num_edges]

            num_nodes = x.size(0)
            node_counts.append(num_nodes)

            # Create dense edge features
            # edge_feature_dim = 1  # Edge features are scalar
            # dense_edge_features = torch.zeros((num_nodes, num_nodes, edge_feature_dim), dtype=edge_attr.dtype)
            # dense_edge_features[edge_index[0], edge_index[1], 0] = edge_attr

            # Create dense edge features in one-hot format
            dense_edge_features = torch.zeros((num_nodes, num_nodes, max_edge_type), dtype=torch.float)
            for i, (src, tgt) in enumerate(edge_index.t()):
                dense_edge_features[src, tgt, edge_attr[i]] = 1.0

            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            # Apply pre_transform if defined
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            dense_edge_features_list.append(dense_edge_features)
            # dense_edge_features_onehot_list.append(dense_edge_features_onehot)

            pbar.update(1)


        pbar.close()

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save(data, osp.join(self.processed_dir, f'{self.split}_data.pt'))
        torch.save(slices, osp.join(self.processed_dir, f'{self.split}_slices.pt'))
        torch.save(node_counts, osp.join(self.processed_dir, f'{self.split}_node_counts.pt'))
        torch.save(dense_edge_features_list, osp.join(self.processed_dir, f'{self.split}_dense_edge_features.pt'))


    def _load_processed_data(self):
        self.data, self.slices = self._load_data_slices()
        node_counts_path = osp.join(self.processed_dir, f'{self.split}_node_counts.pt')
        self.node_counts = torch.load(node_counts_path)
        self.dense_edge_features_list = torch.load(osp.join(self.processed_dir, f'{self.split}_dense_edge_features.pt'))


    def _load_data_slices(self):
        data = torch.load(osp.join(self.processed_dir, f'{self.split}_data.pt'))
        slices = torch.load(osp.join(self.processed_dir, f'{self.split}_slices.pt'))
        return data, slices


    def _processed_files_exist(self):
        return all(osp.exists(osp.join(self.processed_dir, f)) for f in self.processed_file_names)


    def len(self):
        return len(self.node_counts)


    def get(self, idx):
        # Get PyG Data object
        data = self._get_graph_data(idx)

        # Add dense edge features
        data.edge_features = self.dense_edge_features_list[idx]

        return data

    def _get_graph_data(self, idx):
        data = Data()
        for key in self.data.keys():

            if key in ['num_nodes']:
                continue

            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()

            if key == 'edge_index':
                data[key] = item[:, start:end]
            else:
                data[key] = item[start:end]

        return data


# class SizeGroupedZINC(ZINC):
#     def __init__(
#         self,
#         root: str,
#         subset: bool = False,
#         split: str = 'train',
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#         pre_filter: Optional[Callable] = None,
#         force_reload: bool = False,
#     ) -> None:
#         """
#         Initialize the SizeGroupedZINC dataset.
#
#         Args:
#             root (str): Root directory where the dataset should be saved.
#             subset (bool, optional): If True, load a subset of the dataset.
#             split (str, optional): One of 'train', 'val', or 'test'.
#             transform (callable, optional): A function/transform that takes in
#                 a Data object and returns a transformed version.
#             pre_transform (callable, optional): A function/transform that takes in
#                 a Data object and returns a transformed version. Applied before
#                 saving to disk.
#             pre_filter (callable, optional): A function that takes in a Data object
#                 and returns a boolean value, indicating whether the data object
#                 should be included in the final dataset.
#             force_reload (bool, optional): Whether to re-process the dataset.
#         """
#         self.split = split  # Store the split for later use
#         super().__init__(
#             root,
#             subset=subset,
#             split=split,
#             transform=transform,
#             pre_transform=pre_transform,
#             pre_filter=pre_filter,
#             force_reload=force_reload
#         )
#
#         # Check for processed data if force_process is False
#         if not self.force_process and self.processed_files_exist():
#             # Load processed data
#             self.load_processed_data()
#         else:
#             # Process the data if it doesn’t exist or if force_process is True
#             self.process()
#
#         # Initialize size_groups as empty
#         self.size_groups = defaultdict(list)
#         self.compute_size_groups()
#
#     @property
#     def edge_features_list(self):
#         """
#         Lazy-load the dense edge features when accessed.
#         """
#         if not hasattr(self, '_edge_features_list') or self._edge_features_list is None:
#             path = osp.join(self.processed_dir, f'{self.split}_dense_edges.pt')
#             if not osp.exists(path):
#                 raise FileNotFoundError(f"Dense edge features file not found at {path}.")
#             self._edge_features_list = torch.load(path)
#             assert len(self) == len(self._edge_features_list), \
#                 f"Mismatch between dataset size ({len(self)}) and dense edge features ({len(self._edge_features_list)})."
#         return self._edge_features_list
#
#     def processed_file_names(self):
#         return [
#             f'{self.split}_data.pt',
#             f'{self.split}_slices.pt',
#             f'{self.split}_node_counts.pt',
#             f'{self.split}_distance_mats.pt'
#         ]
#
#
#     def processed_files_exist(self):
#         """Check if processed files already exist for the split."""
#         return all(os.path.exists(os.path.join(self.processed_dir, file)) for file in self.processed_file_names())
#
#
#     def get(self, idx):
#         """
#         Override the default __getitem__ method to return the graph with dense edge features.
#
#         Args:
#             idx (int): Index of the graph.
#
#         Returns:
#             Data: A PyG Data object with an additional 'edge_features' attribute.
#         """
#         # Retrieve the PyG Data object using the parent class's __getitem__
#         data = super().get(idx)
#
#         # Attach the corresponding dense edge features
#         data.edge_features = self.edge_features_list[idx]
#
#         return data
#
#     def compute_size_groups(self):
#         """
#         Compute node_counts and populate size_groups.
#         Call this method after the dataset is fully loaded.
#         """
#         self.size_groups = defaultdict(list)
#         for idx, data in enumerate(self):
#             self.size_groups[data.num_nodes].append(idx)
#
#     def load_processed_data(self):
#         """Load processed data and node counts."""
#         self.data, self.slices = self.load_data_slices()
#         node_counts_path = os.path.join(self.processed_dir, f'{self.split}_node_counts.pt')
#         self.node_counts = torch.load(node_counts_path)  # number of nodes in each graph, list of integers
#
#         # Load matrix data
#         self.distance_mats = torch.load(os.path.join(self.processed_dir, f'{self.split}_distance_mats.pt'))  # distance tensor for each graph
#         self.affinities = torch.load(os.path.join(self.processed_dir, f'{self.split}_affinities.pt'))
#         self.edge_features_list = torch.load(os.path.join(self.processed_dir, f'{self.split}_edge_features.pt'))  # list of edge features, [[n,n,4], [m,m,4], ...]
#
#
#     def load_data_slices(self):
#         """Load PyG graph data and slices."""
#         data = torch.load(os.path.join(self.processed_dir, f'{self.split}_data.pt'))
#         slices = torch.load(os.path.join(self.processed_dir, f'{self.split}_slices.pt'))
#         return data, slices
#
#     def process(self) -> None:
#         """
#         Process raw data to generate processed data and dense edge features.
#         """
#         for split in ['train', 'val', 'test']:
#             # Load the raw data for the split
#             raw_split_path = osp.join(self.raw_dir, f'{split}.pickle')
#             if not osp.exists(raw_split_path):
#                 raise FileNotFoundError(f"Raw split file not found at {raw_split_path}.")
#
#             with open(raw_split_path, 'rb') as f:
#                 mols = pickle.load(f)
#
#             indices = list(range(len(mols)))
#
#             if self.subset:
#                 split_index_path = osp.join(self.raw_dir, f'{split}.index')
#                 if not osp.exists(split_index_path):
#                     raise FileNotFoundError(f"Index file not found at {split_index_path}.")
#                 with open(split_index_path, 'r') as f:
#                     indices = [int(x) for x in f.read().strip().split(',')]
#
#             pbar = tqdm(total=len(indices), desc=f'Processing {split} dataset')
#
#             data_list = []
#             dense_edge_list = []  # List to store dense edge matrices
#
#             for idx in indices:
#                 mol = mols[idx]
#
#                 # Extract node features and labels
#                 x = mol['atom_type'].to(torch.long).view(-1, 1)  # Shape: [num_nodes, 1]
#                 y = mol['logP_SA_cycle_normalized'].to(torch.float)  # Shape: [1] or scalar
#
#                 # Extract adjacency matrix and edge features
#                 adj = mol['bond_type']  # Assuming shape: [num_nodes, num_nodes]
#                 edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # Shape: [2, num_edges]
#                 edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)  # Shape: [num_edges, ...]
#
#                 # Determine edge feature dimension
#                 edge_feature_dim = edge_attr.shape[1] if edge_attr.dim() > 1 else 1
#
#                 # Create a dense edge feature tensor
#                 num_nodes = x.shape[0]
#                 dense_edge_features = torch.zeros((num_nodes, num_nodes, edge_feature_dim), dtype=edge_attr.dtype)
#
#                 # Populate the dense edge feature tensor
#                 if edge_feature_dim > 1:
#                     for i, (src, tgt) in enumerate(edge_index.t()):
#                         dense_edge_features[src, tgt] = edge_attr[i]
#                 else:
#                     for i, (src, tgt) in enumerate(edge_index.t()):
#                         dense_edge_features[src, tgt, 0] = edge_attr[i]
#
#                 # Create the PyG Data object
#                 data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
#
#                 # Apply pre-filtering and pre-transformations
#                 if self.pre_filter is not None and not self.pre_filter(data):
#                     pbar.update(1)
#                     continue
#
#                 if self.pre_transform is not None:
#                     data = self.pre_transform(data)
#
#                 # Append the data object and the dense edge features
#                 data_list.append(data)
#                 dense_edge_list.append(dense_edge_features)
#                 pbar.update(1)
#
#             pbar.close()
#
#             # Save the processed data
#             processed_split_path = osp.join(self.processed_dir, f'{split}.pt')
#             self.save(data_list, processed_split_path)
#
#             # Save the dense edge features as a list of tensors
#             dense_edges_split_path = osp.join(self.processed_dir, f'{split}_dense_edges.pt')
#             torch.save(dense_edge_list, dense_edges_split_path)
