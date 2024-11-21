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


def ppgn_tensor(pyg_data, dataset_name, add_edge_weight=False):
    """
    Convert PyG DataBatch to tensor efficiently.
    Returns a tensor of shape [batch_size, 19, num_nodes, num_nodes]
    where 19 channels are:
    - 0-12: Node features as diagonal matrices
    - 13: Distance matrix
    - 14: Affinity matrix
    - 15-18: Edge features
    """

    batch_size = pyg_data.ptr.shape[0] - 1  # Get batch size from ptr
    num_nodes = pyg_data.x.shape[0] // batch_size  # Nodes per graph
    device = pyg_data.x.device

    x_features = pyg_data.x.shape[-1]  # Number of node features

    # Initialize output tensor
    depth = x_features + 1 if add_edge_weight else x_features
    if dataset_name == 'qm9_pos':
        depth += 2 + 4  # Add distance, affinity, and edge features
    elif dataset_name == 'zinc':
        depth += 4 # Add edge features
        if hasattr(pyg_data, 'EigVecs'):
            depth += 4 # Add EigVecs
    out = torch.zeros((batch_size, depth, num_nodes, num_nodes), device=device)

    # Handle node features (channels 0-12)
    x_batched = pyg_data.x.view(batch_size, num_nodes, x_features)  # [batch_size, num_nodes, x_features]
    for i in range(x_features):
        # Efficiently create diagonal matrices for each feature across all batches
        out[:, i] = torch.diag_embed(x_batched[..., i])

    if dataset_name == 'qm9_pos':
        assert hasattr(pyg_data, 'x') and hasattr(pyg_data, 'distance_mat') and hasattr(pyg_data, 'affinity') and hasattr(pyg_data, 'edge_features')

        # Reshape distance and affinity matrices
        out[:, 13] = pyg_data.distance_mat.view(batch_size, num_nodes, -1)
        out[:, 14] = pyg_data.affinity.view(batch_size, num_nodes, -1)

        # Handle edge features (channels 15-18)
        # Reshape from [batch*num_nodes, num_nodes, 4] to [batch, num_nodes, num_nodes, 4]
        edge_features = pyg_data.edge_features.view(batch_size, num_nodes, num_nodes, 4)
        # Permute to get [batch, 4, num_nodes, num_nodes] and assign to channels 15-18
        out[:, 15:19] = edge_features.permute(0, 3, 1, 2)

    elif dataset_name == 'zinc':
        assert hasattr(pyg_data, 'x') and hasattr(pyg_data, 'edge_attr') and hasattr(pyg_data, 'num_nodes')

        edge_features = pyg_data.edge_features.view(batch_size, num_nodes, num_nodes, 4)

        if hasattr(pyg_data, 'EigVecs'):
            eigvec_features = pyg_data.EigVecs.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, eigvec_dim]
            eigvec_dim = eigvec_features.shape[-1]  # Number of EigVec channels
            for i in range(eigvec_dim):
                out[:, x_features + i] = torch.diag_embed(eigvec_features[..., i])

            # Permute to get [batch, 4, num_nodes, num_nodes] and assign to channels 15-18
            out[:, 25:29] = edge_features.permute(0, 3, 1, 2)
        else:
            out[:, x_features:x_features+4] = edge_features.permute(0, 3, 1, 2)

    else:
        raise ValueError(f"Dataset name {dataset_name} not supported with PPGN.")

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
            out[:, depth-1] = adj_edge_weight

    return out



# def ppgn_tensor(pyg_data, dataset_name, add_edge_weight=False):
#     """
#     Convert PyG DataBatch to tensor efficiently.
#     Supports QM9 and other datasets like ZINC with varying features and structure.
#     Returns a tensor of shape [batch_size, num_channels, num_nodes, num_nodes].
#     """
#     batch_size = pyg_data.ptr.shape[0] - 1  # Get batch size from ptr
#     num_nodes = pyg_data.x.shape[0] // batch_size  # Nodes per graph
#     device = pyg_data.x.device
#
#     # Determine number of node features
#     x_features = pyg_data.x.shape[-1]
#
#     # Initialize a list to dynamically collect components
#     out_components = []
#
#     # Handle node features as diagonal matrices
#     x_batched = pyg_data.x.view(batch_size, num_nodes, x_features)  # [batch_size, num_nodes, x_features]
#     diag_matrices = torch.diag_embed(x_batched)  # [batch_size, num_nodes, num_nodes, x_features]
#     diag_matrices = diag_matrices.permute(0, 3, 1, 2)  # [batch_size, x_features, num_nodes, num_nodes]
#     out_components.append(diag_matrices)
#
#     # Dataset-specific processing
#     if dataset_name == 'qm9_pos':
#         # Add distance matrix (channel 13)
#         assert hasattr(pyg_data, 'distance_mat'), "QM9 requires 'distance_mat' in the PyG data."
#         distance_mat = pyg_data.distance_mat.view(batch_size, num_nodes, num_nodes)
#         out_components.append(distance_mat.unsqueeze(1))  # Add as single-channel tensor
#
#         # Add affinity matrix (channel 14)
#         assert hasattr(pyg_data, 'affinity'), "QM9 requires 'affinity' in the PyG data."
#         affinity = pyg_data.affinity.view(batch_size, num_nodes, num_nodes)
#         out_components.append(affinity.unsqueeze(1))  # Add as single-channel tensor
#
#         # Add edge features (channels 15-18)
#         assert hasattr(pyg_data, 'edge_features'), "QM9 requires 'edge_features' in the PyG data."
#         edge_features = pyg_data.edge_features.view(batch_size, num_nodes, num_nodes, -1)  # [batch, nodes, nodes, edge_features]
#         edge_features = edge_features.permute(0, 3, 1, 2)  # [batch, edge_features, num_nodes, num_nodes]
#         out_components.append(edge_features)
#
#     elif dataset_name == 'zinc':
#         # Add edge attributes (number of features may vary)
#         assert hasattr(pyg_data, 'edge_attr'), "ZINC requires 'edge_attr' in the PyG data."
#         edge_features = pyg_data.edge_attr.view(batch_size, num_nodes, num_nodes, -1)  # [batch, nodes, nodes, edge_features]
#         edge_features = edge_features.permute(0, 3, 1, 2)  # [batch, edge_features, num_nodes, num_nodes]
#         out_components.append(edge_features)
#
#         # Optional: Add EigVecs if available
#         if hasattr(pyg_data, 'EigVecs'):
#             eig_vecs = pyg_data.EigVecs.view(batch_size, num_nodes, num_nodes, -1)
#             eig_vecs = eig_vecs.permute(0, 3, 1, 2)  # [batch, eig_features, num_nodes, num_nodes]
#             out_components.append(eig_vecs)
#
#     else:
#         raise ValueError(f"Dataset name {dataset_name} not supported with PPGN.")
#
#     # Add edge weights (optional)
#     if add_edge_weight and hasattr(pyg_data, 'edge_weight') and pyg_data.edge_weight is not None:
#         adj_edge_weight = torch.zeros((batch_size, num_nodes, num_nodes), device=device)
#
#         # Assign edge weights to adjacency matrices
#         edge_batch = pyg_data.batch[pyg_data.edge_index[0]]  # Assign edges based on source node
#         src_global = pyg_data.edge_index[0]  # [num_edges_total]
#         tgt_global = pyg_data.edge_index[1]  # [num_edges_total]
#         ptr = pyg_data.ptr
#         batch_node_start = ptr[edge_batch]
#         src_rel = src_global - batch_node_start
#         tgt_rel = tgt_global - batch_node_start
#         adj_edge_weight[edge_batch, src_rel, tgt_rel] = pyg_data.edge_weight
#         out_components.append(adj_edge_weight.unsqueeze(1))  # Add as single-channel tensor
#
#     # Concatenate components along the channel dimension
#     out = torch.cat(out_components, dim=1)  # [batch_size, num_channels, num_nodes, num_nodes]
#     return out

