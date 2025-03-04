from utils.tunner_img import train_func
import traceback
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import os
import torch
import argparse
import numpy as np

# Create argument parser
parser = argparse.ArgumentParser(description="Train model with given parameters")
parser.add_argument("--data_dir", type=str, default=None, help="path to data dir")
parser.add_argument(
    "--max_epochs", type=int, default=250, help="Number of epochs to train the model"
)
parser.add_argument(
    "--n_samples", type=int, default=3, help="Number of tuning samples"
)

def main(args):
    # data_dir = os.path.join(os.getcwd(), "data")
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else os.path.join(os.getcwd(), "data")
    )
    save_dir = os.path.join(os.getcwd(), "tune_img", "ray_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    class_weights = [
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
    class_weights = torch.from_numpy(np.array(class_weights)).float()
    config = {
        "mode": "img",
        "data_dir": data_dir,
        "learning_rate": 1e-3, #tune.choice([1e-3, 5e-4, 1e-4, 5e-4, 1e-5]),
        "batch_size": 64,
        "optimizer": "adamW", #tune.choice(["adam", "sgd", "adamW"]),
        "epochs": args.max_epochs,
        "gpus": torch.cuda.device_count(),
        "use_mf": False, #tune.choice([True, False]),
        "spatial_attention": False, #tune.choice([True, False]),
        "use_residual": True, #tune.choice([True, False]),
        "n_classes": 9,
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "resolution": tune.choice(["10m", "20m", "10m_bilinear"]),
        "scheduler": "asha",  # tune.choice(["plateau", "steplr", "cosine"]),
        "transforms": "random", #tune.choice(["random", "compose", "None"]),
        "save_dir": save_dir,
        "n_samples": args.n_samples,
        "season": "4seasons", #tune.choice(["summer", "fall", "2seasons", "4seasons", "all"]),
        "loss": "rwmse", #tune.choice(["mse", "mae", "wmse", "rwmse", "kl"]),
        "leading_loss": False, #tune.choice([True, False]),
        "weighted_loss": False, #tune.choice([True, False]),
        "remove_bands": tune.choice([True, False]),
    }
    config["prop_weights"] = class_weights if config["weighted_loss"] else torch.ones(9)
    try:
        # wandb.init(project='M3F-Net-ray')
        scheduler = ASHAScheduler(max_t=100, grace_period=10, reduction_factor=3)
        trainable_with_gpu = tune.with_resources(
            train_func, {"gpu": config.get("gpus", 1)}
        )
        tuner = tune.Tuner(
            trainable_with_gpu,
            tune_config=tune.TuneConfig(
                metric="val_r2",
                mode="max",
                scheduler=scheduler,
                num_samples=config["n_samples"],
            ),
            run_config=train.RunConfig(
                storage_path=config["save_dir"],
                log_to_file=("my_stdout.log", "my_stderr.log"),
                callbacks=[
                    WandbLoggerCallback(
                        project="SRS-Net",
                        group="v2",
                        api_key=os.environ["WANDB_API_KEY"],
                        log_config=True,
                        save_checkpoints=True,
                    )
                ],
            ),
            param_space=config,
        )
        results = tuner.fit()
        print(
            "Best trial config: {}".format(
                results.get_best_result("val_r2", "max").config
            )
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    params = parser.parse_args()
    main(params)
