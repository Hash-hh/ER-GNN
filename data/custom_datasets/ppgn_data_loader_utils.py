import random
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import Sampler


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
    batch_data = Batch.from_data_list(batch)
    return batch_data


def create_same_size_dataloader(dataset, batch_size, shuffle, num_workers=0):
    """Create a dataloader that returns batches of same-size graphs."""
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