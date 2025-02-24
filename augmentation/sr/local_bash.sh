#!/bin/bash

# Define the root directory
root_dir="/mnt/d/rfm_sen"

# Define the seasons
seasons=("spring" "summer" "fall" "winter")

# Iterate over each season
for season in "${seasons[@]}"; do
  # Define the input directory
  input_dir="$root_dir/$season/L2A_S"

  # Define the output directory
  output_dir="$root_dir/$season/composites_10m"

  # Create the output directory if it doesn't exist
  mkdir -p "$output_dir"

  # Find all MTD_MSIL2A.xml files in the .SAFE directories
  find "$input_dir" -type f -name "MTD_MSIL2A.xml" | while read -r xml_file; do
    # Extract the base name of the directory containing the .xml file
    base_name=$(basename "$(dirname "$xml_file")")

    # Extract the time and location from the file path
    IFS='_' read -r -a parts <<< "$base_name"

    # Time is the 3rd element (index 2)
    time="${parts[2]}"

    # Location is the 5th element (index 4)
    location="${parts[5]}"

    # Combine location and time to form the output file name
    output_file="$output_dir/${location}_${time}.tif"
    
    # Run the Python script
    python EIFFEL_Sen2_SR_Predict.py --input "$xml_file" --output "$output_file"
  done
done
