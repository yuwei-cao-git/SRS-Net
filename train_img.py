import argparse
from utils.trainer_img import train
import os
import wandb
import torch
import numpy as np


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train model with given parameters")

    # Add arguments
    parser.add_argument(
        "--task", type=str, default="regression", choices=["regression", "classify"]
    )
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--n_bands", type=int, default=9, help="number bands per tile")
    parser.add_argument("--n_classes", type=int, default=9, help="number classes")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="initial learning rate"
    )
    parser.add_argument("--optimizer", type=str, default="adamW", help="optimizer")
    parser.add_argument("--scheduler", type=str, default="steplr", help="scheduler")
    parser.add_argument("--log_name", type=str, required=True, help="Log file name")
    parser.add_argument(
        "--resolution",
        type=str,
        choices=[
            "20m",
            "10m",
            "10m_bilinear",
            "10m_bilinear_split",
            "5m_bilinear_split",
        ],
        default="20m",
        help="Resolution to use for the data",
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--network",
        default="resunet",
        choices=["resunet", "unet"],
        help="Use residual connections (set flag to enable)",
    )
    parser.add_argument(
        "--fusion_mode",
        default="stack",
        choices=["stack", "sf", "mf", "cs_mf"],
        help="Use fuse modes in seasonal feature fusion",
    )
    parser.add_argument("--transforms", default="random")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument(
        "--season", default="4seasons", type=str, help="season(s) for training"
    )
    parser.add_argument("--loss", default="wrmse")
    parser.add_argument(
        "--leading_loss",
        action="store_true",
    )
    parser.add_argument("--remove_bands", action="store_true")

    # Parse arguments
    params = vars(parser.parse_args())
    params["data_dir"] = (
        params["data_dir"]
        if params["data_dir"] is not None
        else "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed"
    )
    params["save_dir"] = os.path.join(os.getcwd(), "img_logs", params["log_name"])
    prop_weights = [
        0.13429631,
        0.02357711,
        0.05467328,
        0.04353036,
        0.02462899,
        0.03230562,
        0.2605792,
        0.00621396,
        0.42019516,
    ]
    prop_weights = torch.from_numpy(np.array(prop_weights)).float()
    params["prop_weights"] = (
        prop_weights if params["loss"].startswith("w") else torch.ones(9)
    )

    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    print(params)

    wandb.init(project="SRS-Net", group="v3")
    # Call the train function with parsed arguments
    train(params)


if __name__ == "__main__":
    main()
