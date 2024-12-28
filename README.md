# SRS-Net: Super-Resolution Sentinel-2 for Tree Species Composition Estimation
This repository provides tools and workflows for leveraging super-resolution techniques on Sentinel-2 satellite imagery to enhance tree species composition estimation. The project integrates cutting-edge deep learning approaches with forestry research to improve the granularity and accuracy of forest inventory data.

## Features
- Super-Resolution Sentinel-2 Data: Enhance Sentinel-2 imagery to higher spatial resolution for detailed analysis.
- Tree Species Estimation: Utilize state-of-the-art models to estimate species composition in forest stands.
- Open Science Workflow: Scripts and models designed for reproducibility and adaptability to new datasets.

## Repository Structure
|-- data/  
|   |-- raw/              # Original Sentinel-2 imagery  
|   |-- processed/        # Preprocessed and super-resolved imagery  
|-- models/  
|   |-- super_resolution/ # Models for super-resolution  
|   |-- TSC_estimation/ # Models for tree species composition  
|-- notebooks/  
|   |-- data_preparation.ipynb    # Steps to preprocess Sentinel-2 data  
|   |-- training_workflow.ipynb   # Model training and evaluation  
|-- utils/  
|   |-- visualization.py          # Plotting and visualization tools  
|-- README.md  

## Installation
1. Clone the repository:
```
git clone https://github.com/yuwei-cao-git/SRS-Net.git  
cd SRS-Net  
```
2. Install dependencies:
```
pip install -r requirements.txt  
```
Dependencies
```
> Python 3.8+
> Lightning
> Geopandas
> Rasterio
> Scikit-learn
> GDAL
```
## Workflow Overview
1. Preprocessing Sentinel-2 Data
- Prepare Sentinel-2 images for super-resolution.
- Georeference and validate the imagery for downstream analysis.
2. Super-Resolution Application
- Use the provided models to upscale Sentinel-2 imagery to higher spatial resolution.
3. Tree Species Composition Estimation
- Train or use pretrained models to estimate species composition in forest stands.
- Generate maps and summaries of species distribution.

## Example Use Case
Run the data_preparation.ipynb notebook to preprocess Sentinel-2 data. Then use the training_workflow.ipynb notebook to train a model for tree species composition estimation.

## License
This project is licensed under the MIT License.
