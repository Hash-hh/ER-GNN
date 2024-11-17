from data.custom_datasets.qm9_size_grouped_fast import create_same_size_dataloader
from data.custom_datasets.qm9_size_grouped_fast import SizeGroupedQM9
from torch.utils.data import Subset
import os

data_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(data_path, 'datasets', 'QM9_pos')

# dataset = CustomQM9Dataset(root=data_path, split='train')
# batch_size = 4
# dataloader = create_custom_dataloader(dataset, batch_size, shuffle=True)
#
# for batch in dataloader:
#     x = batch.x
#     edge_index = batch.edge_index
#     distance_mats = batch.distance_mats
#     affinities = batch.affinities
#     edge_features = batch.edge_features
#     # Proceed with training or validation
#
#     print(len(batch))

# dataset = SizeGroupedQM9(root=data_path, split='train', force_process=False)
batch_size = 32
dataset = SizeGroupedQM9(root=data_path, split='test')
dataset._data.y = dataset._data.y[:, 0: 0 + 1]
# subset_indices = list(range(16))
# train_set = dataset.get_subset(subset_indices)
# train_set = dataset[:16]
train_set = dataset

dataloader = create_same_size_dataloader(dataset=train_set, batch_size=batch_size, shuffle=True)

for batch in dataloader:

    x = batch.x
    edge_index = batch.edge_index
    distance_mats = batch.distance_mat
    affinities = batch.affinity
    edge_features = batch.edge_features

    print("x size: ", x.size())
    print("edge_index size: ", edge_index.size())
    print("distance_mats size: ", distance_mats.size())
    print("affinities size: ", affinities.size())
    print("edge_features size: ", edge_features.size())

    print(len(batch))