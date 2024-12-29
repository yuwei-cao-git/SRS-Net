import os
from pathlib import Path
from itertools import cycle, islice
import torch
from torch_scatter import scatter_mean
import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from plyer import notification
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


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