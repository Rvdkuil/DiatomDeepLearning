##############################################################################
## Code to copy merged virtual slidescans made in LAS X to a new folder and ##
## split them for further processing                                        ##
##############################################################################
# Import libraries
import os
import shutil
import re
from split_image import split_image
from PIL import Image

# Copy the slides
# Define source and destination paths
to_count_dir = 'E:/to_count'
all_scans_dir = 'E:/all_scans'

# Create destination folder if it doesn't exist
os.makedirs(all_scans_dir, exist_ok=True)
os.chdir(all_scans_dir)

# Function to check if file (with or without _0, _1 suffix) exists
def file_already_exists(base_name, destination):
    pattern = re.compile(re.escape(base_name) + r'(_\d+)?\.tif$', re.IGNORECASE)
    for f in os.listdir(destination):
        if pattern.fullmatch(f):
            return True
    return False

# Go through each subfolder in the to_count directory
for subfolder in os.listdir(to_count_dir):
    subfolder_path = os.path.join(to_count_dir, subfolder)
    if os.path.isdir(subfolder_path):
        original_file = os.path.join(subfolder_path, 'TileScan_001_Merging.tif')
        if os.path.isfile(original_file):
            new_name = f"{subfolder}.tif"
            if not file_already_exists(subfolder, all_scans_dir):
                dest_file = os.path.join(all_scans_dir, new_name)
                shutil.copy2(original_file, dest_file)
                print(f"Copied and renamed: {original_file} → {dest_file}")
            else:
                print(f"Skipped (already exists): {subfolder}")

#%%
# Split the slides into two columns
# Disable image size safety limit
Image.MAX_IMAGE_PIXELS = None

# Path to your folder
image_dir = "E:/All_scans"

# Get all .tif files in the folder
all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.tif')]

# Exclude files ending in _0.tif or _1.tif
pattern = re.compile(r'.*_[01]\.tif$', re.IGNORECASE)
files_to_process = [f for f in all_files if not pattern.match(f)]

# Run split_image for each matching file
for filename in files_to_process:
    image_path = os.path.join(image_dir, filename)
    
    try:
        print(f"Splitting: {filename}")
        split_image(image_path, 1, 2, should_cleanup=True, should_square=False)
    except Exception as e:
        print(f"Error processing {filename}: {e}")