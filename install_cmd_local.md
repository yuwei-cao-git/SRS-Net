mamba create --name tf -y python=3.8
mamba activate tf
python -m pip install --upgrade pip
pip3 install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 'numpy<2' 'setuptools<70'
pip3 install pointnext==0.0.5 mamba-ssm[causal-conv1d]==2.2.2
mamba install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
mamba deactivate
mamba activate tf
mamba install protobuf==3.19.6
mamba install seaborn
mamba install typing_extensions
mamba install tensorflow-gpu=2.2.0
mamba install keras
mamba install gdal scikit-image imageio matplotlib
conda install gdal
mamba install scikit-image
mamba install imageio
mamba install matplotlib
mamba install geotiff
mamba install proj
mamba install rasterio
mamba install dos2unix
dos2unix local_bash.sh
mamba install tqdm
cd /mnt/d/Sync/research/tree\ species\ estimation/code/fusion/M3F-Net/augmentation/EIFFEL_Sentinel2_SR/
./local_bash.sh