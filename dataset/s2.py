import os
import rasterio
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from os.path import join


def load_tile_names(file_path):
    """
    Load tile names from a .txt file.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        tile_names (list): List of tile names.
    """
    with open(file_path, "r") as f:
        tile_names = f.read().splitlines()
    return tile_names


class TreeSpeciesDataset(Dataset):
    def __init__(self, tile_names, processed_dir, datasets, mode):
        """
        Args:
            tile_names (list): List of tile filenames to load.
            processed_dir (str): Base directory containing the processed data folders.
            datasets (list): List of dataset folder names to include (e.g., ['s2/spring', 's2/summer',...]).
        """
        self.mode = mode
        self.tile_names = tile_names
        self.processed_dir = processed_dir
        self.datasets = datasets  # List of dataset folder names

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, idx):
        if self.mode == "img":
            tile_name = self.tile_names[idx].split(" ")[0] + ".tif"
        else:
            tile_name = self.tile_names[idx]
        input_data_list = []

        # Load data from each dataset (spring, summer, fall, winter, etc.)
        for dataset in self.datasets:
            dataset_path = os.path.join(self.processed_dir, dataset, tile_name)
            with rasterio.open(dataset_path) as src:
                input_data = src.read()  # Read the bands (num_bands, H, W)
                input_data_list.append(
                    torch.from_numpy(input_data).float()
                )  # Append each season's tensor to the list

        # Load the corresponding label (target species composition)
        label_path = os.path.join(self.processed_dir, "labels/tiles_128", tile_name)

        with rasterio.open(label_path) as src:
            target_data = src.read()  # (num_bands, H, W)
            nodata_value_label = src.nodata  # NoData value for the labels

            # Create a NoData mask for the target data
            if nodata_value_label is not None:
                mask = np.any(
                    target_data == nodata_value_label, axis=0
                )  # Collapse bands to (H, W)
            else:
                mask = np.zeros_like(
                    target_data[0], dtype=bool
                )  # Assume all valid if no NoData value

        # Convert the target and mask to PyTorch tensors
        target_tensor = torch.from_numpy(
            target_data
        ).float()  # Shape: (num_output_channels, H, W)
        mask_tensor = torch.from_numpy(mask).bool()  # Shape: (H, W)

        # Return the list of input tensors for each season, the target tensor, and the mask tensor
        return input_data_list, target_tensor, mask_tensor


class TreeSpeciesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        Args:
            tile_names (dict): Dictionary with 'train', 'val', and 'test' keys containing lists of tile filenames to load.
            processed_dir (str): Directory where processed data is located.
            datasets_to_use (list): List of dataset names to include (e.g., ['s2/spring', 's2/summer', ...]).
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.config = config
        # Tile names for train, validation, and test
        # User specifies which datasets to use
        self.processed_dir = join(config["data_dir"], f'{self.config["resolution"]}m')
        self.tile_names = {
            "train": load_tile_names(
                join(self.processed_dir, "dataset/train_tiles.txt")
            ),
            "val": load_tile_names(join(self.processed_dir, "dataset/val_tiles.txt")),
            "test": load_tile_names(join(self.processed_dir, "dataset/test_tiles.txt")),
        }
        self.datasets_to_use = [
            "rmf_s2/spring/tiles_128",
            "rmf_s2/summer/tiles_128",
            "rmf_s2/fall/tiles_128",
            "rmf_s2/winter/tiles_128",
        ]

        self.batch_size = config["batch_size"]

    def setup(self, stage=None):
        """
        Sets up the dataset for train, validation, and test splits.
        """
        # Create datasets for train, validation, and test
        self.train_dataset = TreeSpeciesDataset(
            self.tile_names["train"],
            self.processed_dir,
            self.datasets_to_use,
            self.config["mode"],
        )
        self.val_dataset = TreeSpeciesDataset(
            self.tile_names["val"],
            self.processed_dir,
            self.datasets_to_use,
            self.config["mode"],
        )
        self.test_dataset = TreeSpeciesDataset(
            self.tile_names["test"],
            self.processed_dir,
            self.datasets_to_use,
            self.config["mode"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )
