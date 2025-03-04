import os
import rasterio
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from os.path import join
import torchvision.transforms as transforms


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
    def __init__(self, tile_names, processed_dir, datasets, resolution, augment="None", remove_bands=False):
        """
        Args:
            tile_names (list): List of tile filenames to load.
            processed_dir (str): Base directory containing the processed data folders.
            datasets (list): List of dataset folder names to include (e.g., ['s2/spring', 's2/summer',...]).
        """
        self.tile_names = tile_names
        self.processed_dir = processed_dir
        self.datasets = datasets  # List of dataset folder names
        self.augment = augment
        self.resolution = resolution
        self.remove_bands = remove_bands

        # Define the transformation pipeline
        size = 128 if self.resolution != "10m_bilinear" else 256
        if self.augment == "compose":
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.RandomAffine(
                        degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                    ),
                ]
            )
        elif self.augment == "random":
            self.transform = transforms.RandomApply(
                torch.nn.ModuleList(
                    [
                        transforms.RandomCrop(size=(size, size)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                        transforms.RandomRotation(degrees=(0, 180)),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                        ),
                    ]
                ),
                p=0.3,
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, idx):
        # tile_name = self.tile_names[idx].split(" ")[0] + ".tif"
        tile_name = self.tile_names[idx]
        if not tile_name.endswith(".tif"):
            tile_name = tile_name.strip()
            tile_name += ".tif"
        input_data_list = []

        # Load data from each dataset (spring, summer, fall, winter, etc.)
        for dataset in self.datasets:
            dataset_path = os.path.join(self.processed_dir, dataset, tile_name)
            with rasterio.open(dataset_path) as src:
                if self.remove_bands:
                    # Define bands to remove (1-based [1, 9, 10])
                    bands_to_remove = {1, 9, 10}
                    # Generate list of all valid 1-based band indices to read
                    bands_to_read = [b for b in range(1, src.count + 1) if b not in bands_to_remove]
                    # Read only the needed bands
                    input_data = src.read(bands_to_read)
                else:
                    input_data = src.read()  # Read the bands (num_bands, H, W)
                tensor_data = torch.from_numpy(input_data).float()
                # Apply augmentation if enabled
                if self.transform:
                    tensor_data = self.transform(tensor_data)
                input_data_list.append(
                    tensor_data
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
        self.processed_dir = join(config["data_dir"], f'{self.config["resolution"]}')
        self.resolution = config["resolution"]
        self.remove_bands = config["remove_bands"]
        self.tile_names = {
            "train": load_tile_names(
                join(self.processed_dir, "dataset/train_tiles.txt")
            ),
            "val": load_tile_names(join(self.processed_dir, "dataset/val_tiles.txt")),
            "test": load_tile_names(join(self.processed_dir, "dataset/test_tiles.txt")),
        }

        if self.config["season"] == "2seasons":
            self.datasets_to_use = [
                "rmf_s2/summer/tiles_128",
                "rmf_s2/fall/tiles_128",
            ]
        elif self.config["season"] == "4seasons":
            self.datasets_to_use = [
                "rmf_s2/spring/tiles_128",
                "rmf_s2/summer/tiles_128",
                "rmf_s2/fall/tiles_128",
                "rmf_s2/winter/tiles_128",
            ]
        elif self.config["season"] == "all":
            self.datasets_to_use = [
                "rmf_s2/spring/tiles_128",
                "rmf_s2/summer/tiles_128",
                "rmf_s2/fall/tiles_128",
                "rmf_s2/winter/tiles_128",
                "rmf_phenology/tiles_128",  # 36 bands
                "rmf_spl_climate/tiles_128",  # 1 band
                "rmf_spl_dem/tiles_128",  # 1 band
            ]
        else:
            self.datasets_to_use = [
                f"rmf_s2/{self.config['season']}/tiles_128",
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
            self.resolution,
            augment=None,
            remove_bands=self.remove_bands
        )
        if self.config["transforms"] != "None":
            aug_train_dataset = TreeSpeciesDataset(
                self.tile_names["train"],
                self.processed_dir,
                self.datasets_to_use,
                self.resolution,
                augment=self.config["transforms"],
                remove_bands=self.remove_bands
            )
            self.train_dataset = torch.utils.data.ConcatDataset(
                [self.train_dataset, aug_train_dataset]
            )
        self.val_dataset = TreeSpeciesDataset(
            self.tile_names["val"],
            self.processed_dir,
            self.datasets_to_use,
            self.resolution,
            remove_bands=self.remove_bands
        )
        self.test_dataset = TreeSpeciesDataset(
            self.tile_names["test"],
            self.processed_dir,
            self.datasets_to_use,
            self.resolution,
            remove_bands=self.remove_bands
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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )
