from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune, train
from pytorch_lightning.loggers import WandbLogger
from models.s2_model import Model
import os
import wandb
import time
from dataset.s2 import TreeSpeciesDataModule
from .common import generate_eva


def train_func(config):
    seed_everything(1)

    wandb_logger = WandbLogger(
        project="SRS-Net",
        group="v0",
        name=f"trial_{tune.Trainable().trial_id}",
        save_dir=config["save_dir"],
        log_model=True,
    )
    wandb_logger.experiment.config.update(config)

    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(config)

    # Call setup explicitly to initialize datasets
    data_module.setup(stage="fit")

    # Use the calculated input channels from the DataModule to initialize the model
    model = Model(config)

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_r2",  # Track the validation loss
        filename="final_model",
        save_top_k=1,  # Only save the best model
        mode="min",  # We want to minimize the validation loss
    )

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        logger=wandb_logger,
        devices=config.get("gpus", 1),
        num_nodes=1,
        strategy="ddp",
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, data_module)

    # using a pandas DataFrame to recode best results
    if model.best_test_outputs is not None:
        output_dir = os.path.join(
            config["save_dir"],
            f"trial_{tune.Trainable().trial_id}",
            "outputs",
        )
        _ = generate_eva(model, config["classes"], output_dir)
        # wandb_logger.log_text(key="preds", dataframe=sp_df) # WARNING:root:Truncating wandb.Table object to 200000 rows.

    # Report the final metric to Ray Tune
    final_result = trainer.callback_metrics["val_r2"].item()
    train.report({"val_r2": final_result})

    # Test the model after training
    trainer.test(model, data_module)

    time.sleep(5)  # Wait for wandb to finish logging
    wandb.finish()
