from pytorch_lightning import Trainer
from models.s2_model import Model
from dataset.s2 import TreeSpeciesDataModule
import os
import os
import glob
import rasterio
from rasterio.merge import merge


def test(config):
    chk_dir = os.path.join("img_logs", config["log_name"], "checkpoints", "final_model.ckpt")

    # Initialize the DataModule
    data_module = TreeSpeciesDataModule(config)

    # 1. Setup the DataModule for testing (e.g., test dataset)
    data_module.setup(stage="test")

    # 2. Load the best model from checkpoint
    litmodel = Model.load_from_checkpoint(chk_dir, vis=config["vis_mode"])

    # 3. Create a PyTorch Lightning Trainer for testing
    trainer = Trainer(
        devices=1,
        num_nodes=1,
    )

    # 4. Test the model
    trainer.test(litmodel, data_module)
    print("Testing complete.")


def vis(prediction_folder, nodata_value=255, output_path="merged_predictions.tif"):
    """
    Merge all prediction tiles in `prediction_folder` into a single raster.
    
    Args:
        prediction_folder (str): Path to folder containing prediction GeoTIFFs.
        nodata_value (float or int): NoData value to set in the merged raster.
        output_path (str): File path to save the merged raster.
    """
    # Find all .tif files
    tile_paths = glob.glob(os.path.join(prediction_folder, "*.tif"))
    if len(tile_paths) == 0:
        raise ValueError(f"No TIFF files found in {prediction_folder}")
    
    # Open all tile datasets
    src_files_to_mosaic = [rasterio.open(path) for path in tile_paths]

    # Merge them
    mosaic, out_transform = merge(src_files_to_mosaic, nodata=nodata_value)

    # Use metadata from one of the tiles
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "nodata": nodata_value,
    })

    # Save the merged file
    with rasterio.open(os.path.join(prediction_folder, output_path), "w", **out_meta) as dest:
        dest.write(mosaic)
        
    # Close datasets and optionally delete tile files
    for src in src_files_to_mosaic:
        src.close()

    for path in tile_paths:
        os.remove(path)
    print(f"Removed {len(tile_paths)} tile(s) from: {prediction_folder}")

    print(f"Merged raster saved to: {output_path}")

    
if __name__ == "__main__":
    configs = {
        "task": "classify",
        "batch_size": 32,
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
        "data_dir": "/mnt/e/rmf_img",
        "fusion_mode": "stack",
        "leading_loss": False,
        "learning_rate": 0.001,
        "log_name": "lsc_resunet_wkl_10m_bi_summer_lr5e4_cosinewarm",
        "loss": "mse",
        "n_bands": 9,
        "n_classes": 9,
        "network": "unet",
        "optimizer": "adamW",
        "prop_weights": [0.126, 0.018, 0.055, 0.043, 0.041, 0.228, 0.021, 0.005, 0.461],
        "remove_bands": False,
        "resolution": "10m_bilinear_split",
        "save_dir": "/mnt/d/Sync/research/tree_species_estimation/code/image/SRS-Net/img_logs",
        "scheduler": "cosinewarmup",
        "season": "4seasons",
        "task": "regression",
        "transforms": "random",
        "season": "4seasons",
        "vis_mode": True,
    }
    test(config=configs)
    
    if configs["vis_mode"]:
        prediction_folder=os.path.join("img_logs", configs["log_name"], "outputs/predictions")
        vis(prediction_folder)