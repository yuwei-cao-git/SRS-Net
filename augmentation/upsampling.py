import os
import rasterio
from rasterio.enums import Resampling
from joblib import Parallel, delayed

# bilinear sampling
# Define paths
base_dir = "/mnt/e/rmf_img/20m"
output_base_dir = "/mnt/e/rmf_img/10m_bilinear"


def upsample_tif(input_path, output_path):
    """Upsamples a 20m resolution TIFF file to 10m and saves it."""
    with rasterio.open(input_path) as src:
        # Compute new transform and dimensions
        transform = src.transform * src.transform.scale(
            (src.width / (src.width * 2)), (src.height / (src.height * 2))
        )
        new_width = src.width * 2
        new_height = src.height * 2

        # Read and resample
        data = src.read(
            out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear
        )

        # Update metadata
        profile = src.profile
        profile.update(
            {"height": new_height, "width": new_width, "transform": transform}
        )

        # Save output
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    print(f"Upsampled {input_path} -> {output_path}")


# Collect all .tif files to process
tif_files = []

for root, dirs, files in os.walk(base_dir):
    if "tiles_128" in root:
        rel_path = os.path.relpath(root, base_dir)  # Get relative path
        output_dir = os.path.join(output_base_dir, rel_path)  # Define output folder
        os.makedirs(output_dir, exist_ok=True)  # Create output directory

        for file in files:
            if file.endswith(".tif"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                tif_files.append((input_path, output_path))

# Use joblib to parallelize processing
Parallel(n_jobs=-1)(
    delayed(upsample_tif)(input_path, output_path)
    for input_path, output_path in tif_files
)
