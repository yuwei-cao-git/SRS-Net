from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from models.s2_model import Model
from dataset.s2 import TreeSpeciesDataModule
import os


def train(config):
    seed_everything(1)

    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(config)

    # Call setup explicitly to initialize datasets
    data_module.setup(stage="fit")

    # Use the calculated input channels from the DataModule to initialize the model
    model = Model(config)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
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

    wandb_logger = WandbLogger(name=config["log_name"])

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["epochs"],
        logger=[wandb_logger],
        callbacks=[early_stopping, checkpoint_callback],
        devices=config["gpus"],
        num_nodes=1,
        strategy="ddp",
    )
    # Train the model
    trainer.fit(model, data_module)

    # Save the best model after training
    # trainer.save_checkpoint(os.path.join(config["save_dir"], "final_model.pt"))

    # Test the model after training
    trainer.test(model, data_module)

    # Load the saved model
    # model = UNetLightning.load_from_checkpoint("final_model.ckpt")
