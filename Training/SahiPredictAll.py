##########################################################################
## Code for using SAHI and YOLO to draw inference on a folder of images ##
##########################################################################

# Import libraries
from sahi import AutoDetectionModel
import os
import pandas as pd
from collections import defaultdict
from sahi.predict import get_sliced_prediction
import pickle
import re

# Set working directory to the directory with the virtual slidescans
# Set path to the trained model
wd = "E:/All_scans"
os.chdir(wd)
trained_model = "E:/train153/weights/best.pt"
#%%
# Create functions to append the species counts to a dataframe
def count_species(annotations):
    species_counts = defaultdict(int)
    
    for annotation in annotations:
        species_name = annotation['category_name']
        species_counts[species_name] += 1
    
    return species_counts

def annotations_to_dataframe(annotations):
    # Get species counts
    species_counts = count_species(annotations)
    
    # Convert counts dictionary to DataFrame
    df = pd.DataFrame(list(species_counts.items()), columns=['Species', 'Count'])
    
    return df

# Instantiate a YOLOv11 model for object detection
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=trained_model,
    confidence_threshold=0.80,
    device=0
    )

pictures = os.listdir(wd)
#%%
# Check if the folder contains existing data files. Resume inference or 
# create new data files and start from scratch.

try:
    with open('data.pkl', 'rb') as f:
        species_counts = pickle.load(f)
except FileNotFoundError:
    print("No existing data.pkl file found. Starting fresh.")
    species_counts = {}

try:
    with open('data_original.pkl', 'rb') as f:
        species_counts_original = pickle.load(f)
except FileNotFoundError:
    print("No existing data_original.pkl file found. Starting fresh.")
    species_counts_original = {}

#%%
# Keep track of processed files
processed_files = set(species_counts.keys())

# Perform the inference. Save original species counts and species counts after size-thresholding.
for filename in os.listdir('E:/All_scans'):
    if filename in processed_files:
        print(f"Skipping {filename} as it has already been processed.")
        continue

    try:
        # Perform detection
        result = get_sliced_prediction(
            filename,
            detection_model,
            slice_height=1216,
            slice_width=1216,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            postprocess_match_threshold=0.1,
            postprocess_class_agnostic=False,
            postprocess_type='NMS'
        )

        # Convert detection result to COCO annotations
        coco_annotations = result.to_coco_annotations()

        # Save original counts
        df_annotations_original = annotations_to_dataframe(coco_annotations)
        species_counts_original[filename] = df_annotations_original.groupby('Species')['Count'].sum()

        # Apply size thresholds
        size_thresholds = {
            'LIN': (294.5, 294.5),
            'PLA': (189.63, 189.63),
            'EN1': (249.25, 249.25),
            'NI1': (112.75, 112.75)
        }

        filtered_annotations = []
        for ann in coco_annotations:
            try:
                category = ann['category_name']
                x, y, w, h = ann['bbox']
                if category in size_thresholds:
                    min_w, min_h = size_thresholds[category]
                    if w <= min_w and h <= min_h:
                        continue
                filtered_annotations.append(ann)
            except:
                continue

        # Save filtered counts
        df_annotations_filtered = annotations_to_dataframe(filtered_annotations)
        species_counts[filename] = df_annotations_filtered.groupby('Species')['Count'].sum()

        # Mark file as processed
        processed_files.add(filename)

        # Save progress for both files
        with open('data.pkl', 'wb') as f:
            pickle.dump(species_counts, f)
        with open('data_original.pkl', 'wb') as f:
            pickle.dump(species_counts_original, f)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All files processed.")
#%%
# Create a dataframe from the species_counts dictionary
final_df = pd.DataFrame(species_counts).fillna(0).reset_index()
final_df.rename(columns={'index': 'Species'}, inplace=True)

print(final_df)

#%%
# Add the species counts for both halves of every slide
df = final_df.set_index('Species')  

# Identify column pairs to sum
columns_to_combine = [(df.columns[i], df.columns[i + 1]) for i in range(0, len(df.columns) - 1, 2)]

# Iterate through each column pair and sum values
for col1, col2 in columns_to_combine:
    new_col_name = col1  # New column name will be the name of the first column in the pair
    df[new_col_name] = df[col1] + df[col2]

# Drop the original columns that have been summed
df.drop(columns=[col2 for col1, col2 in columns_to_combine], inplace=True)

# Reset the index to bring 'Species' back as a regular column
df.reset_index(inplace=True)
#%%
# Sort columns in dataframe by core ID and depth
# Function to extract sorting key
def sorting_key(col_name):
    match = re.match(r'(SB\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)_.*', col_name)
    if match:
        prefix = match.group(1)
        start_range = float(match.group(2))
        end_range = float(match.group(4))
        return (prefix, start_range, end_range)
    else:
        return (col_name, float('inf'), float('inf'))  

# Separate 'species' column and other columns
other_columns = [col for col in df.columns if col != 'Species']

# Sort columns using the sorting key
sorted_columns = sorted(other_columns, key=sorting_key)

# Combine 'species' as the first column
final_columns = ['Species'] + sorted_columns

# Reorder dataframe columns
df_sorted = df[final_columns]
#%%
# Save dataframe to excel
df_sorted.to_excel("E:/InferenceResults.xlsx", index=False)
