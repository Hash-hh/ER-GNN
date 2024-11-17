# import gzip
# import json
# import os
# import numpy as np
# import torch
# import h5py
# from torch_geometric.data import InMemoryDataset, Data, download_url
# from tqdm import tqdm
# import pickle
from collections import defaultdict

import os
import torch
import h5py
import pickle
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import download_url


from torch_geometric.data import InMemoryDataset, Data, download_url
import os
import torch
import h5py
import pickle
from collections import defaultdict
from tqdm import tqdm

class QM9_pos(InMemoryDataset):
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
        # self.size_groups = defaultdict(list)
        # for idx, num_nodes in enumerate(self.node_counts):
        #     self.size_groups[num_nodes].append(idx)

    @property
    def raw_file_names(self):
        return ['QM9_test.p', 'QM9_val.p', 'QM9_train.p']

    @property
    def processed_file_names(self):
        return [
            f'{self.split}_data.pt',
            f'{self.split}_slices.pt',
            f'{self.split}_node_counts.pt',
            'distance_affinity.h5'
        ]

    def processed_files_exist(self):
        """Check if processed files already exist for the split."""
        return all(os.path.exists(os.path.join(self.processed_dir, file)) for file in self.processed_file_names)

    def download(self):
        for split, url in self.url.items():
            filename = f'QM9_{split}.p' if split != 'valid' else 'QM9_val.p'
            download_url(url=url, folder=self.raw_dir, filename=filename)

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

                    # Apply pre_transform if defined
                    if self.pre_transform is not None:
                        pyg_graph = self.pre_transform(pyg_graph)

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
        # with h5py.File(self.h5f_path, 'r') as h5f:
        #     graph_group = h5f[f'{self.split}/graph_{idx}']
        #
        #     # Load and validate matrix data
        #     for key in ['distance_mat', 'affinity', 'edge_features']:
        #         if key in graph_group:
        #             setattr(data, key, torch.tensor(graph_group[key][:]))

        return data

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

# def map_qm9_to_pyg(instance):
#     # Extract features for the current molecule
#     x = torch.tensor(instance['usable_features']['x'], dtype=torch.float)  # Node features
#     # edge_features = torch.tensor(instance['usable_features']['edge_features'], dtype=torch.float)
#     edge_index = torch.tensor(instance['original_features']['edge_index'])
#     pos = torch.tensor(instance['original_features']['pos'], dtype=torch.float)
#     edge_attr = torch.tensor(instance['original_features']['edge_attr'], dtype=torch.float)  # contains the same information (bond type) as edge_features but in a different format
#     y = torch.tensor(instance['y'], dtype=torch.float)  # Labels
#
#     # Only keep fixed-size attributes in Data
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
#     return data
#
#
# class QM9_pos(InMemoryDataset):
#     url = {
#         'test': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAxhVewejSl7gVMACa1tBUda/QM9_test.p?dl=1',
#         'valid': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AAAOfEx-jGC6vvi43fh0tOq6a/QM9_val.p?dl=1',
#         'train': 'https://www.dropbox.com/sh/acvh0sqgnvra53d/AADtx0EMRz5fhUNXaHFipkrza/QM9_train.p?dl=1',
#     }
#
#     def __init__(self,
#                  root,
#                  split,
#                  transform=None, pre_transform=None, pre_filter=None,
#                  force_process=False):
#         assert split in ['train', 'valid', 'test']
#         self.split = split
#         self.force_process = force_process
#         super().__init__(root, transform, pre_transform, pre_filter)
#
#         # Load processed data if available
#         self.slices = torch.load(os.path.join(self.processed_dir, f'{split}_slice.pt'))
#         self.data = torch.load(os.path.join(self.processed_dir, f'{split}_data.pt'))
#
#     @property
#     def raw_file_names(self):
#         return ['QM9_test.p', 'QM9_val.p', 'QM9_train.p']
#
#     @property
#     def processed_file_names(self):
#         return ['train_data.pt', 'valid_data.pt', 'test_data.pt',
#                 'train_slice.pt', 'valid_slice.pt', 'test_slice.pt',
#                 'distance_affinity.h5']  # Adding HDF5 file for matrices
#
#     def download(self):
#         download_url(url=self.url['train'], folder=self.raw_dir, filename='QM9_train.p', log=True)
#         download_url(url=self.url['test'], folder=self.raw_dir, filename='QM9_test.p', log=True)
#         download_url(url=self.url['valid'], folder=self.raw_dir, filename='QM9_valid.p', log=True)
#
#     def process(self):
#         # Check if processing is necessary
#         if not self.force_process and all(
#                 os.path.exists(os.path.join(self.processed_dir, f)) for f in self.processed_file_names):
#             print("Processed files already exist. Skipping processing.")
#             return
#
#         # Create an HDF5 file to store distance and affinity matrices
#         h5_path = os.path.join(self.processed_dir, 'distance_affinity.h5')
#         with h5py.File(h5_path, 'w') as h5f:
#
#             for split in ['train', 'valid', 'test']:  # using val instead of valid because that's how it's downloaded
#                 print(f'Processing {split} split')
#                 file_path = os.path.join(self.raw_dir, f'QM9_{split}.p')
#                 with open(file_path, "rb") as f:
#                     graphs = pickle.load(f)
#
#                     # Group molecules by number of atoms
#                     grouped_graphs = defaultdict(list)  # {num_atoms: [graphs]}
#                     for graph in graphs:
#                         num_atoms = len(graph['usable_features']['x'])
#                         grouped_graphs[num_atoms].append(graph)
#
#                     pyg_graphs = []
#
#                     # Process each group of molecules with the same atom count
#                     for num_atoms, group in grouped_graphs.items():
#                         grp = h5f.create_group(f'{split}/{num_atoms}_atoms')
#
#                         for i, graph in enumerate(tqdm(group)):
#                             # Convert graph to PyTorch Geometric format
#                             pyg_graph = map_qm9_to_pyg(graph)
#                             if self.pre_transform is not None:
#                                 pyg_graph = self.pre_transform(pyg_graph)
#                             pyg_graphs.append(pyg_graph)
#
#                             # Store distance and affinity matrices in HDF5
#                             grp.create_dataset(f'molecule_{i}/distance_mat', data=graph['usable_features']['distance_mat'])
#                             grp.create_dataset(f'molecule_{i}/affinity', data=graph['usable_features']['affinity'])
#                             grp.create_dataset(f'molecule_{i}/edge_features', data=graph['usable_features']['edge_features'])
#
#                         if self.pre_filter is not None:
#                             pyg_graphs = [d for d in pyg_graphs if self.pre_filter(d)]
#
#                 # Collate and save Data and Slices
#                 d, s = self.collate(pyg_graphs)
#                 torch.save(d, os.path.join(self.processed_dir, f'{split}_data.pt'))
#                 torch.save(s, os.path.join(self.processed_dir, f'{split}_slice.pt'))
