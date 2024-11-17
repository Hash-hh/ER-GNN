import numpy as np
import torch

# def ppgn_tensor(pyg_data):
#     """Convert PyG Data object to tensor."""
#
#     # x = pyg_data.x
#     # distance_mat = pyg_data.distance_mat
#     # affinity = pyg_data.affinity
#     # edge_features = pyg_data.edge_features
#     # # edge_index = pyg_data.edge_index
#     # # y = pyg_data.y
#     # # edge_candidate = pyg_data.edge_candidate
#     # # num_edge_candidate = pyg_data.num_edge_candidate
#     # # pos = pyg_data.pos
#     # # edge_attr = pyg_data.edge_attr
#
#     # tensor_graph = []
#     # for instance in pyg_data.to_data_list():
#     #     nodes_num = instance.x.shape[0]
#     #     graph = np.empty((nodes_num, nodes_num, 19))
#     #     for i in range(13):
#     #         # 13 features per node - for each, create a diag matrix of it as a feature
#     #         graph[:, :, i] = np.diag(instance.x[:, i])
#     #     graph[:, :, 13] = instance.distance_mat
#     #     graph[:, :, 14] = instance.affinity
#     #     graph[:, :, 15:] = instance.edge_features  # shape n x n x 4
#     #     tensor_graph.append(graph)
#     # tensor_graph = np.array(tensor_graph, dtype="object")
#     # for i in range(tensor_graph.shape[0]):
#     #     tensor_graph[i] = np.transpose(tensor_graph[i], [2, 0, 1])
#
#     tensor_graph = []
#     for instance in pyg_data.to_data_list():
#         nodes_num = instance.x.shape[0]
#         # Initialize a tensor with zeros for the full graph representation
#         graph = torch.zeros((19, nodes_num, nodes_num), device=instance.x.device)
#
#         # Populate diagonal features (13 node features as diagonal matrices)
#         for i in range(13):
#             graph[i, :, :] = torch.diag(instance.x[:, i])
#
#         # Assign distance matrix and affinity matrix as separate layers
#         graph[13, :, :] = instance.distance_mat
#         graph[14, :, :] = instance.affinity
#
#         # Assign edge features (already n x n x 4) directly into the last four layers
#         graph[15:19, :, :] = instance.edge_features.permute(2, 0, 1)
#
#         # Append the completed graph tensor to the list
#         tensor_graph.append(graph)
#
#     # Stack all graphs into a single tensor for batch processing
#     tensor_graph = torch.stack(tensor_graph)

    # return tensor_graph


import torch


def ppgn_tensor(pyg_data, add_edge_weight=False):
    """
    Convert PyG DataBatch to tensor efficiently.
    Returns a tensor of shape [batch_size, 19, num_nodes, num_nodes]
    where 19 channels are:
    - 0-12: Node features as diagonal matrices
    - 13: Distance matrix
    - 14: Affinity matrix
    - 15-18: Edge features
    """

    # assert that the pyg_data contains the necessary attributes
    assert hasattr(pyg_data, 'x') and hasattr(pyg_data, 'distance_mat') and hasattr(pyg_data, 'affinity') and hasattr(pyg_data, 'edge_features')

    batch_size = pyg_data.ptr.shape[0] - 1  # Get batch size from ptr
    num_nodes = pyg_data.x.shape[0] // batch_size  # Nodes per graph
    device = pyg_data.x.device

    # Initialize output tensor
    depth = 20 if add_edge_weight else 19
    out = torch.zeros((batch_size, depth, num_nodes, num_nodes), device=device)

    # Handle node features (channels 0-12)
    x_batched = pyg_data.x.view(batch_size, num_nodes, 13)  # [batch_size, num_nodes, 13]
    for i in range(13):
        # Efficiently create diagonal matrices for each feature across all batches
        out[:, i] = torch.diag_embed(x_batched[..., i])

    # Reshape distance and affinity matrices
    out[:, 13] = pyg_data.distance_mat.view(batch_size, num_nodes, -1)
    out[:, 14] = pyg_data.affinity.view(batch_size, num_nodes, -1)

    # Handle edge features (channels 15-18)
    # Reshape from [batch*num_nodes, num_nodes, 4] to [batch, num_nodes, num_nodes, 4]
    edge_features = pyg_data.edge_features.view(batch_size, num_nodes, num_nodes, 4)
    # Permute to get [batch, 4, num_nodes, num_nodes] and assign to channels 15-18
    out[:, 15:19] = edge_features.permute(0, 3, 1, 2)

    if add_edge_weight:
        if hasattr(pyg_data, 'edge_weight') and pyg_data.edge_weight is not None:
            # Initialize adjacency matrices for edge weights
            adj_edge_weight = torch.zeros((batch_size, num_nodes, num_nodes), device=device)

            # Compute edge batch assignments
            edge_batch = pyg_data.batch[pyg_data.edge_index[0]]  # Assign edges based on source node

            # Compute global node indices for edges
            src_global = pyg_data.edge_index[0]  # [num_edges_total]
            tgt_global = pyg_data.edge_index[1]  # [num_edges_total]

            # Get start indices of nodes for each graph in the batch
            ptr = pyg_data.ptr  # [batch_size + 1]
            batch_node_start = ptr[edge_batch]  # Start index for nodes in each graph

            # Compute relative node indices within each graph
            src_rel = src_global - batch_node_start  # [num_edges_total]
            tgt_rel = tgt_global - batch_node_start  # [num_edges_total]

            # Assign edge weights to adjacency matrices
            adj_edge_weight[edge_batch, src_rel, tgt_rel] = pyg_data.edge_weight

            # Assign to output tensor
            out[:, 19] = adj_edge_weight

    return out

