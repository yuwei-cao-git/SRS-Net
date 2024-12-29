#!/bin/bash

# Load python module, and additional required modules
module purge 
#module load gcc/9.3.0 arrow python/3.10 scipy-stack/2022a
module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index ray[tune]
pip install --no-index tensorboardX lightning pytorch_lightning torch torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas 
pip install --no-index imageio rasterio
pip install laspy[laszip]
#pip install -r requirements.txt