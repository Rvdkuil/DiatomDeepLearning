from sahi import AutoDetectionModel
import os
import pandas as pd
from collections import defaultdict
from sahi.predict import get_sliced_prediction
import pickle
import re
os.chdir("D:/all_scans")
#%%
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

# Instantiate a YOLOv8 model for object detection
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="C:/Users/rvdku/Desktop/Biologie/Palaeostage/Report/Exceldata/best(2).pt",
    confidence_threshold=0.80,
    device=0
    )

pictures = os.listdir('D:/all_scans')
#species_counts = {}
#%%
with open('data.pkl', 'rb') as f:
    species_counts = pickle.load(f)
# Set to keep track of processed filenames
processed_files = set(species_counts.keys())
processed_files = set()
#%%

for filename in os.listdir('D:/all_scans'):
    
    if filename in processed_files:
        print(f"Skipping {filename} as it has already been processed.")
        continue  # Skip this filename if it has already been processed
    
    try:
        # Perform detection on the image file
        result = get_sliced_prediction(
            filename,
            detection_model,
            slice_height=1216,
            slice_width=1216,
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4,
            postprocess_match_threshold=0.05,
            postprocess_class_agnostic = True
        )
        
        # Convert detection result to COCO annotations
        coco_annotations = result.to_coco_annotations()
        
        # Convert annotations to DataFrame
        df_annotations = annotations_to_dataframe(coco_annotations)
        
        # Group species counts by filename in the dictionary
        species_counts[filename] = df_annotations.groupby('Species')['Count'].sum()
        
        # Mark filename as processed
        processed_files.add(filename)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All files processed.")()
with open('data.pkl', 'wb') as f:
    pickle.dump(species_counts, f)
#%%
# Create a dataframe from the species_counts dictionary
final_df = pd.DataFrame(species_counts).fillna(0).reset_index()
final_df.rename(columns={'index': 'Species'}, inplace=True)

print(final_df)

#%%
df = final_df.set_index('Species')  # Assuming 'Species' is the index column
first_numeric_column = df.columns[0]
df = df.drop(columns=[first_numeric_column])
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
# Function to extract sorting key
def sorting_key(col_name):
    match = re.match(r'(SB\d+)_(\d+(\.\d+)?)-(\d+(\.\d+)?)_.*', col_name)
    if match:
        prefix = match.group(1)
        start_range = float(match.group(2))
        end_range = float(match.group(4))
        return (prefix, start_range, end_range)
    else:
        return (col_name, float('inf'), float('inf'))  # Handle unexpected formats

# Separate 'species' column and other columns
other_columns = [col for col in df.columns if col != 'Species']

# Sort columns using the sorting key
sorted_columns = sorted(other_columns, key=sorting_key)

# Combine 'species' as the first column
final_columns = ['Species'] + sorted_columns

# Reorder dataframe columns
df_sorted = df[final_columns]
#%%
df_sorted.to_excel("D:/reallastinference.xlsx", index=False)