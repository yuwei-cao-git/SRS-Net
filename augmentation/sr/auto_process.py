#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
import shutil
import subprocess
import zipfile

from tqdm import tqdm


# In[6]:


def unzip_directory(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(in_dir)
    for item in tqdm(
        os.listdir(in_dir), desc="Unzipping Folders", leave=True, colour="red"
    ):
        if item.endswith(".zip"):
            file_name = os.path.abspath(item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(out_dir)
            zip_ref.close()


# In[10]:


def run_l2a_process(xml_file, output_file):
    # Check if output folder exists, create if necessary
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Define the L2A_Process command
    l2a_process_cmd = f'python D:\Sync\research\tree_species_estimation\code\fusion\M3F_Net\augmentation\EIFFEL_Sentinel2_SR\EIFFEL_Sen2_SR_Predict.py --input {xml_file} --output {output_file}'
    print ("Running...", l2a_process_cmd)
    os.system(l2a_process_cmd) 
    
    # Run L2A_Process command
    try:
        subprocess.run(l2a_process_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running L2A_Process: {e}")


# In[4]:

directory = r"E:\rmf_s2"
for season in ["spring"]:
    input_folder = f"{directory}/{season}"
    l2a_folder = f"{input_folder}/L2A"

    safe_folders = [
        f"{l2a_folder}/{file}"
        for file in os.listdir(l2a_folder)
        if file.endswith(".SAFE")
    ]
    output_folder = f"{input_folder}/composite_10m"
    for safe_folder in tqdm(
        safe_folders, desc="Super resolution prediction processing", leave=True, colour="green"
    ):
        tile = os.path.basename(safe_folder).split("_")[2]
        date = os.path.basename(safe_folder).split("_")[-2]
        xml_file = os.path.join(safe_folder, 'MTD_MSIL2A.xml')
        run_l2a_process(safe_folder, os.path.join(output_folder, f"{tile}_{date}_10m.tif"))


# %%