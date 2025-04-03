from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from dataset.s2 import TreeSpeciesDataModule
import pandas as pd
import matplotlib.pyplot as plt


def train(config):
    seed_everything(123)

    wandb_logger = WandbLogger(
        project="SRS-Net",
        group="v3",
        name=config["log_name"],
        save_dir=config["save_dir"],
    )

    csv_logger = CSVLogger(
        save_dir=config["save_dir"],
    )

    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(config)

    # Use the calculated input channels from the DataModule to initialize the model
    if config["task"] == "regression":
        from models.s2_model import Model
    else:
        from models.s2_leading_species import Model

    model = Model(config)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        mode="min",  # Set "min" for validation loss
        verbose=True,
    )

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Track the validation loss
        filename="final_model",
        dirpath=config["save_dir"],
        save_top_k=1,  # Only save the best model
        mode="min",  # We want to minimize the validation loss
    )

    # Add LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval="step")  # or 'epoch'

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        logger=[wandb_logger, csv_logger],
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        num_nodes=1,
        devices="auto",
        strategy="ddp",
    )

    # Train the model
    trainer.fit(model, data_module)

    metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    aggreg_metrics = []

    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss_epoch", "val_loss_epoch"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.savefig(f"{csv_logger.log_dir}/loss.png")

    # Test the model after training
    trainer.test(model, data_module)
