#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --gpus-per-node=4            # number of gpus per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --tasks-per-node=4 # This is the number of model replicas we will place on the GPU.
#SBATCH --mem=128G
#SBATCH --job-name="multi-gpu-srs-train"
#SBATCH --time=00:30:00        # Specify run time 

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
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -xf $project/SRS-Net/data/20m.tar -C ./data
tar -xf $project/SRS-Net/data/10m_bilinear.tar -C ./data
tar -xf $project/SRS-Net/data/10m.tar -C ./data
echo "Data transfered"

# Load python module, and additional required modules
echo "load modules"
module load python StdEnv gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch==2.5.0
pip install --no-index ray[tune] tensorboardX lightning pytorch_lightning torchaudio==2.5.0 torchdata torcheval torchmetrics torchtext torchvision==0.20.0 rasterio imageio wandb numpy pandas
pip install --no-index scikit-learn seaborn
pip install --no-index mamba-ssm

echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
wandb login *

# Run python script
# Define your resolution list
for resolution in "20m" "10m" "10m_bilinear"
do
    # Create output directory
    mkdir -p ~/scratch/img_logs/${resolution}
    echo "Created output dir: ~/scratch/img_logs/${resolution}"

    # Run your model multiple times
    srun python train_img.py \
        --data_dir './data' \
        --batch_size 64 \
        --use_residual \
        --resolution "$resolution" \
        --log_name "ResUnet_s4_${resolution}"

    # Package logs: adjust paths if logs are saved elsewhere
    cd "$SLURM_TMPDIR"
    tar -cf ~/scratch/img_logs/${resolution}/logs.tar ./img_logs/*
    echo "Logs archived for resolution: ${resolution}"
done

# Check the exit status
if [ $job_failed -ne 0 ]; then
    echo "Job failed"
else
    echo "Job completed successfully."
fi

echo "theend"