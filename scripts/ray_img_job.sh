#!/bin/bash
#SBATCH --job-name=ray_img_tune
#SBATCH --output=img_tune_%j.out
#SBATCH --error=img_tune_%j.err
#SBATCH --time=20:00:00        # Specify run time 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

next_output_dir=$(date +%Y%m%d%H%M%S)
mkdir ~/scratch/tune_img/${next_output_dir}
echo "created output dir"

# Trap the exit status of the job
trap 'job_failed=$?' EXIT

# code transfer
cd $SLURM_TMPDIR
mkdir work
cd work
git clone git@github.com:yuwei-cao-git/SRS-Net.git
cd SRS-Net
echo "Source code cloned!"

# data transfer
mkdir -p data/10m
mkdir -p data/20m
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/SRS-Net/data/10m.tar -C ./data/10m
tar -xf $project/SRS-Net/data/20m.tar -C ./data/20m
echo "Data transfered"

# Load python module, and additional required modules
module purge 
module load python StdEnv gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index ray[all]
pip install --no-index ray[tune] tensorboardX lightning pytorch_lightning torch torchaudio torchdata torcheval torchmetrics torchtext torchvision rasterio imageio wandb numpy pandas
pip install --no-index scikit-learn seaborn
# pip install -r requirements.txt

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

export WANDB_API_KEY=*
wandb login
#Run python script
echo "Start runing model.................................................................................."
srun python tune_img.py

cd $SLURM_TMPDIR
tar -cf ~/scratch/tune_img/${next_output_dir}/logs.tar ./tune_img/ray_results/*
tar -xf ~/scratch/tune_img/${next_output_dir}/logs.tar -C ~/scratch/tune_img/{next_output_dir}
rm -r ./tune_img/ray_results/
# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed, deleting directory: ${next_output_dir}"
    # rm -r "${next_output_dir}"
else
    echo "Job completed successfully."
fi

echo "theend"
