###############################################################################
## k-fold cross-validation as adapted from the Ultralytics website and       ##
## https://www.kaggle.com/code/tataganesh/k-fold-cross-validation-and-yolov8 ##
###############################################################################

# Import libraries
import shutil
from pathlib import Path
from collections import Counter
from IPython.display import clear_output
import json
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
import glob, os
import subprocess

# Set paths
annotations_directory = "D:/sorted_images/combi"
kfold_base_path = Path("D:/sorted_images/combi/k_fold")

#%%
# Preprocess annotations. Change labels of species with <30 instances to "other"
def process_annotations(directory, min_instances=30):
    label_counts = {}  # Dictionary for counts of each species label
    processed_label_counts = {}  # Dictionary for counts of species labels after processing
    processed_dir = os.path.join(directory, "processed_annotations")
    
    os.makedirs(processed_dir, exist_ok=True)  # Ensure output directory exists

    print("Processing annotation files...")

    # Count labels and process files
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue  # Skip non-JSON files

        filepath = os.path.join(directory, filename)
        processed_filepath = os.path.join(processed_dir, filename)

        try:
            with open(filepath, "r") as file:
                data = json.load(file)
            
            # Count initial label occurrences
            for shape in data.get("shapes", []):
                label = shape.get("label")
                label_counts[label] = label_counts.get(label, 0) + 1

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue  # Skip this file and move to the next

    # Process files and update labels
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(directory, filename)
        processed_filepath = os.path.join(processed_dir, filename)

        try:
            with open(filepath, "r") as file:
                data = json.load(file)

            for shape in data.get("shapes", []):
                original_label = shape["label"]
                if label_counts.get(original_label, 0) < min_instances:
                    shape["label"] = "other"  # Change label if it appears < min_instances

                # Update processed label counts
                new_label = shape["label"]
                processed_label_counts[new_label] = processed_label_counts.get(new_label, 0) + 1

            # Save modified annotations
            with open(processed_filepath, "w") as processed_file:
                json.dump(data, processed_file, indent=4)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return label_counts, processed_label_counts, processed_dir  # Return processed directory path

# Run the processing function
initial_counts, processed_counts, processed_dir = process_annotations(annotations_directory)

# Print initial summary
print("\nInitial Label Summary:")
for label, count in initial_counts.items():
    print(f"{label}: {count} instances")

# Print processed summary
print("\nProcessed Label Summary:")
for label, count in processed_counts.items():
    print(f"{label}: {count} instances")

#%%
# Convert labelme annotations to the format accepted by YOLO
def convert_to_yolo(processed_dir):
    """Runs the labelme2yolo command silently (no output)."""
    
    if not os.path.exists(processed_dir):
        print(f"Error: The directory {processed_dir} does not exist.")
        return

    command = [
        "labelme2yolo",
        "--json_dir", processed_dir,
        "--val_size", "0",
        "--output_format", "polygon"
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print("Error: labelme2yolo command failed.")

# Run the conversion
convert_to_yolo(processed_dir)

#%%
# Move label files and image files to the same directory and update the filepaths
dataset_path = os.path.join(processed_dir, "YOLODataset")  # Base dataset path
images_directory = os.path.join(dataset_path, "images")
labels_directory = os.path.join(dataset_path, "labels")
yaml_file_path = os.path.join(dataset_path, "dataset.yaml")

def move_files_and_cleanup(base_directory):
    # Define subdirectories to handle moving files for both images and labels
    subfolders = ["train", "val"]

    # Move images from "train" and "val" subdirectories to "images" directory
    for subfolder in subfolders:
        subfolder_path = os.path.join(images_directory, subfolder)
        if os.path.exists(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    shutil.move(file_path, images_directory)
            shutil.rmtree(subfolder_path)

    # Copy label files from "labels/train" and "labels/val" to both "images" and "labels" directories
    for subfolder in subfolders:
        subfolder_path = os.path.join(labels_directory, subfolder)
        if os.path.exists(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if filename.endswith(".txt") and os.path.isfile(file_path):
                    # Copy label file to images directory
                    shutil.copy(file_path, images_directory)
                    # Copy label file to labels directory (though they are already there)
                    shutil.copy(file_path, labels_directory)
            shutil.rmtree(subfolder_path)

    # Update the YAML file paths
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            yaml_data = yaml.safe_load(file)

        # Updating paths for images in YAML file
        yaml_data["train"] = images_directory.replace("//", "/")
        yaml_data["val"] = os.path.join(images_directory, "val").replace("//", "/")

        # Write the updated YAML back to file
        with open(yaml_file_path, "w") as file:
            yaml.safe_dump(yaml_data, file)

# Run the cleanup and file movement
move_files_and_cleanup(dataset_path)

#%%
# Add background images (still need to automate)

#%%
# Store image and label paths for future use
# Get all .png and .tif files
png_files = glob.glob(images_directory + "/*.png")
tif_files = glob.glob(images_directory + "/*.tif")

# Combine the lists
image_paths = png_files + tif_files

# Find all label files
label_paths = glob.glob(images_directory + "/*.txt")

print(f"Found {len(image_paths)} images and {len(label_paths)} labels.")

#%%
# Get label names from the .yaml file
dataset_path = Path(dataset_path)
labels = sorted(dataset_path.rglob("images/*.txt"))  # all data in "labels"

yaml_file = os.path.join(dataset_path, "dataset.yaml") # your data YAML with data directories and names dictionary
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = list(range(len(classes)))
print(list(zip(classes, cls_idx)))

#%%
# Generate a dataframe to calculate the label distribution
indx = [Path(img).stem for img in image_paths]  # use base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

# Create the label vector for stratification
y = []

# Create a set of label stems for quick lookup
label_stems = {label.stem for label in labels}

# Separate labeled and unlabeled images
labeled_images = []
unlabeled_images = []

for img_path in image_paths:
    img_stem = Path(img_path).stem
    
    if img_stem in label_stems:
        labeled_images.append(img_path)
        label_file = next(label for label in labels if label.stem == img_stem)
        lbl_counter = Counter()
        
        with open(label_file, "r") as lf:
            lines = lf.readlines()

        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(" ")[0])] += 1

        # Use the first class in the label file for stratification
        if lines:
            first_class = int(lines[0].split(" ")[0])
            y.append(first_class)
        else:
            y.append(-1)  # Handle empty label files if necessary

        labels_df.loc[img_stem] = lbl_counter
    else:
        unlabeled_images.append(img_path)
        # Fill the DataFrame with zeros or any default value for the unlabeled images
        labels_df.loc[img_stem] = [0] * len(cls_idx)

labels_df = labels_df.fillna(0.0)  # replace nan values with 0.0
print(labels_df)

#%%
# Create the training and validation splits
# Convert y to numpy array
y = np.array(y)

# use sklearn to perform the StratitfiedKFold
ksplit = 5
kf = StratifiedKFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

labeled_indices = [indx.index(Path(img).stem) for img in labeled_images]
kfolds = []

for train_index, test_index in kf.split(np.zeros(len(labeled_indices)), y):
    train_indices = [labeled_indices[i] for i in train_index]
    test_indices = [labeled_indices[i] for i in test_index]
    
    # Add all unlabeled .tif image indices to the training set
    unlabeled_indices = [
        indx.index(Path(img).stem) 
        for img in unlabeled_images 
        if img.endswith(".tif") and Path(img).stem in indx
    ]
    train_indices += unlabeled_indices
    
    kfolds.append((train_indices, test_indices))

#%%
# Get the label distribution in each split
folds = [f"split_{n}" for n in range(1, ksplit + 1)]
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1E-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio

lbl_distribution = fold_lbl_distrb
print(lbl_distribution)

#%%
# Create .yaml and .txt files for each fold
kfold_base_path = Path("D:/Report/Training_dataset/k_fold")
shutil.rmtree(kfold_base_path) if kfold_base_path.is_dir() else None # Remove existing folder
os.makedirs(str(kfold_base_path)) # Create new folder
yaml_paths = list()
train_txt_paths = list()
val_txt_paths = list()
for i, (train_idx, val_idx) in enumerate(kfolds):
    # Get image paths for train-val split
    train_paths = [image_paths[j] for j in train_idx]
    val_paths = [image_paths[j] for j in val_idx]

    # Create text files to store image paths
    train_txt = kfold_base_path / f"train_{i}.txt"
    val_txt = kfold_base_path / f"val_{i}.txt"

    # Write image paths for training and validation in split i
    with open(str(train_txt), "w") as f:
        f.writelines(s + "\n" for s in train_paths)
    with open(str(val_txt), "w") as f:
        f.writelines(s + "\n" for s in val_paths)

    train_txt_paths.append(str(train_txt))
    val_txt_paths.append(str(val_txt))

    # Create yaml file
    yaml_path = kfold_base_path / f"data_{i}.yaml"
    with open(yaml_path, "w") as ds_y:
        yaml.safe_dump({
            "train": str(train_txt),
            "val": str(val_txt),
            "nc": len(classes),
            "names": classes
        }, ds_y)
    yaml_paths.append(str(yaml_path))
print("Yaml Paths")
print(yaml_paths)

#%%
# Train the model K times. Append results to a dataframe. 
os.environ["WANDB_DISABLED"] = "true"

batch = 6
project = "kfold_cross_validation"
epochs = 1
patience = 500

results = list()

for i in range(ksplit):
    model = YOLO("yolov8s-seg.pt")
    dataset_yaml = yaml_paths[i]
    print(f"Training for fold={i} using {dataset_yaml}")
    model.train(data=dataset_yaml, batch=batch, project=project, epochs=epochs, verbose=True, plots=True, patience=patience, device=0)
    result = model.metrics # Metrics on validation set
    results.append(result) # save output metrics for further analysis
    clear_output()


