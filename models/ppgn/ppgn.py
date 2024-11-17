import torch
import torch.nn as nn
import models.ppgn.layers as layers
import models.ppgn.modules as modules


class PPGN(nn.Module):
    def __init__(self, use_new_suffix, block_features, num_classes, depth_of_mlp, only_downstream):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.use_new_suffix = use_new_suffix  # True or False
        self.block_features = block_features  # List of number of features in each regular block
        original_features_num = 19 if only_downstream else 2*20  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(last_layer_features, next_layer_features, depth_of_mlp)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2*output_features, num_classes, activation_fn=None)
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, num_classes, activation_fn=None))

    def forward(self, input, rewired_data=None):

        if rewired_data is not None:  # if True, concatenate a and b before passing to the network
            rewring_samples = rewired_data.size(0) // input.size(0)
            repeated_data = input.repeat(rewring_samples, 1, 1, 1)
            x = torch.cat([repeated_data, rewired_data], dim=1)
        else:
            x = input

        scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

            if self.use_new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.use_new_suffix:
            # old suffix
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        return scores
