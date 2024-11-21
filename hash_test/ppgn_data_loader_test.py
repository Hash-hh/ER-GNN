from sympy.strategies.branch import debug

from data.custom_datasets.ppgn_data_loader_utils import create_same_size_dataloader
from data.custom_datasets.qm9_size_grouped_fast import SizeGroupedQM9
from torch.utils.data import Subset
import os

data_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(data_path, 'datasets', 'QM9_pos')

batch_size = 32
dataset = SizeGroupedQM9(root=data_path, split='test', debug=True)
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