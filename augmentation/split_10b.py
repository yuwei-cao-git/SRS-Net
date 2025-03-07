import os
import rasterio
from rasterio.windows import Window
from joblib import Parallel, delayed
from tqdm import tqdm

# Define paths
base_dir = "/mnt/e/rmf_img/10m_bilinear"
split_output_dir = "/mnt/e/rmf_img/10m_bilinear_split"
os.makedirs(split_output_dir, exist_ok=True)

# Define dataset splits
splits = ["train", "test", "val"]
split_files = {
    split: os.path.join(base_dir, "dataset", f"{split}_tiles.txt") for split in splits
}
split_output_dirs = {
    split: os.path.join(split_output_dir, "dataset", split) for split in splits
}

# Ensure output directories exist
for split_dir in split_output_dirs.values():
    os.makedirs(split_dir, exist_ok=True)


# Function to split a single TIFF file
def split_tif(input_path, output_dir, base_name):
    """Splits a (256, 256, 9) image into four (128, 128, 9) tiles and saves them."""
    with rasterio.open(input_path) as src:
        # Define the four tile positions
        tile_positions = {
            "00": Window(0, 0, 128, 128),  # Top-left
            "01": Window(128, 0, 128, 128),  # Top-right
            "10": Window(0, 128, 128, 128),  # Bottom-left
            "11": Window(128, 128, 128, 128),  # Bottom-right
        }

        tile_names = []
        for key, window in tile_positions.items():
            tile_filename = f"{base_name}_{key}.tif"
            tile_output_path = os.path.join(output_dir, tile_filename)

            # Read and save the tile
            tile_data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                {
                    "width": 128,
                    "height": 128,
                    "transform": rasterio.windows.transform(window, src.transform),
                }
            )

            with rasterio.open(tile_output_path, "w", **profile) as dst:
                dst.write(tile_data)

            tile_names.append(
                tile_filename.replace(".tif", "")
            )  # Store name without extension

    return tile_names


# Read original train/test/val splits
split_images = {split: set() for split in splits}

for split, file_path in split_files.items():
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            split_images[split] = {
                line.strip() for line in f.readlines()
            }  # Use set for fast lookup

# Collect upsampled images and determine their split
tasks = []
split_results = {split: [] for split in splits}

for root, _, files in os.walk(base_dir):
    if "tiles_128" in root:  # Ensures we are in a dataset's tile folder
        rel_path = os.path.relpath(
            root, base_dir
        )  # Relative path to dataset (e.g., "s2/spring/tiles_128")
        output_dir = os.path.join(
            split_output_dir, rel_path
        )  # Ensure same structure in output
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.endswith(".tif"):
                input_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]

                # Determine the split
                for split in splits:
                    if base_name in split_images[split]:
                        tasks.append((input_path, output_dir, base_name, split))
                        break  # Stop checking once split is found

# Parallel processing of TIFF splitting
split_tiles = Parallel(n_jobs=-1)(
    delayed(split_tif)(input_path, output_dir, base_name)
    for input_path, output_dir, base_name, _ in tasks
)

# Collect results into correct split files
for (_, _, base_name, split), tiles in zip(tasks, split_tiles):
    split_results[split].extend(tiles)

# Save updated train/test/val tile lists
for split in splits:
    split_file = os.path.join(split_output_dir, "dataset", f"{split}_tiles.txt")
    with open(split_file, "w") as f:
        for tile_name in tqdm(split_results[split]):
            f.write(f"{tile_name}\n")

    print(f"Updated {split}_tiles.txt with {len(split_results[split])} tiles.")

print("Dataset split completed with train/test/val maintained.")
