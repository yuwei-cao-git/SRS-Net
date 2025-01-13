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
    parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
    parser.add_argument("--n_bands", type=int, default=9, help="number bands per tile")
    parser.add_argument("--n_classes", type=int, default=9, help="number classes")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument("--optimizer", type=str, default="adamW", help="optimizer")
    parser.add_argument("--scheduler", type=str, default="steplr", help="scheduler")
    parser.add_argument("--log_name", type=str, required=True, help="Log file name")
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[20, 10],
        default=20,
        help="Resolution to use for the data",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--use_mf", action="store_true", help="Use multi-fusion (set flag to enable)"
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Use residual connections (set flag to enable)",
    )
    parser.add_argument(
        "--spatial_attention",
        action="store_true",
        help="Use spatial attention in mf fusion",
    )
    parser.add_argument("--transforms", default="compose")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--season", default="4seasons", help="season(s) for training")
    parser.add_argument("--loss", default="mse")
    parser.add_argument(
        "--leading_loss",
        action="store_true",
    )
    parser.add_argument("--weighted_loss", default=True)

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
    params["prop_weights"] = prop_weights if params["weighted_loss"] else torch.ones(9)

    if not os.path.exists(params["save_dir"]):
        os.makedirs(params["save_dir"])
    print(params)

    wandb.init(project="SRS-Net")
    # Call the train function with parsed arguments
    train(params)


if __name__ == "__main__":
    main()
