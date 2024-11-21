from data.custom_datasets.qm9_pos import QM9_pos
import os
import h5py
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch


# data path is the current directory
data_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(data_path, 'datasets', 'QM9_pos')

dataset = QM9_pos(data_path,
              split='test',
              transform=None,
              pre_transform=None,
                  )

print(dataset)


def custom_collate(batch):
    """
    Custom collate function that handles empty `edge_index` by replacing
    it with a single dummy edge to ensure consistent concatenation.

    Args:
        batch (list): List of Data objects (individual graphs).

    Returns:
        Batch: Batched Data object with `edge_index_list` storing each graph's `edge_index`.
    """
    for data in batch:
        if data.edge_index.size(1) == 0:  # Check for empty edge_index
            # Replace empty edge_index with a single dummy edge
            data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # Standard PyG batch collation, but keep edge_index separately
    batch_data = Batch.from_data_list(batch)

    # Store individual edge_index tensors for each graph in a list
    batch_data.edge_index_list = [data.edge_index for data in batch]

    return batch_data


# Update DataLoader to use custom collate function
loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)

# Loop through batches
# Usage
for batch in loader:
    print(batch)
    # break




# try:
#     h_path = os.path.join(data_path, 'processed', 'distance_affinity.h5')
#     with h5py.File(h_path, 'r') as h5f:
#         print("File opened successfully.")
#         # Perform any read operations
#         distance_matrix = h5f['test/molecule_0/distance_mat'][:]
#         affinity = h5f['test/molecule_0/affinity'][:]
#
#         print(distance_matrix)
#
# except KeyError as e:
#     print(f"KeyError encountered: {e}")
# except Exception as e:
#     print(f"An error occurred: {e}")

