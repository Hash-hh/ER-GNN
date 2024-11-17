from typing import Callable
from models.model_utils import ppgn_tensor

import torch


class HybridModel(torch.nn.Module):
    def __init__(self,
                 upstream: torch.nn.Module,
                 downstream: torch.nn.Module,
                 rewiring: Callable):
        super(HybridModel, self).__init__()
        self.upstream = upstream
        self.downstream = downstream
        self.rewiring = rewiring

    def forward(self, data):
        """
        Forward pass for the model, handling both downstream and rewiring cases.
        """
        # Determine if PPGN processing is required
        is_ppgn = hasattr(data, 'distance_mat')
        add_edge_weight = self.upstream is not None or self.rewiring is not None

        # Process data into tensor format if PPGN is used
        down_data = ppgn_tensor(data, add_edge_weight=add_edge_weight) if is_ppgn else data

        # If no upstream or rewiring, only use downstream model
        if self.upstream is None and self.rewiring is None:
            pred = self.downstream(down_data, rewired_data=None)
            return pred, [data], 0  # No rewiring, no auxiliary loss

        # Handle upstream and rewiring logic
        select_edge_candidates, delete_edge_candidates, edge_candidate_idx = self.upstream(data)
        new_data, auxloss = self.rewiring(
            data,
            select_edge_candidates,
            delete_edge_candidates,
            edge_candidate_idx
        )

        # Convert rewired data into PPGN tensor format if needed
        if is_ppgn:
            assert hasattr(new_data[0], 'edge_weight') and new_data[0].edge_weight is not None, \
                "Rewired data must have edge weights for PPGN."
            new_data_down = ppgn_tensor(new_data[0], add_edge_weight=True)
        else:
            new_data_down = new_data

        # Use downstream model with rewired data
        pred = self.downstream(down_data, new_data_down)
        return pred, new_data, auxloss

