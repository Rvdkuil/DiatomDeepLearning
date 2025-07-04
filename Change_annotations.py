##################################
## Remove and group annotations ##
##################################

# Import libraries

import os
import json
import pandas as pd

# Config
input_folder = "D:/sorted_images/combi"          
excel_file = "D:/Ordered_files_diatoms/paper/ExcelData/Species_names.xlsx"              
sheet_name = "Sheet1"                                     

# Load dataframe from Excel
df = pd.read_excel(excel_file, sheet_name=sheet_name)
df["Delete"] = df["Delete"].astype(bool)

# Labels to delete
delete_labels = set(df[df["Delete"]]["LabelmeCode"])

# Process JSON files
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    original_labels = {shape["label"] for shape in shapes}

    # Delete entire file if all labels are to be deleted
    if original_labels.issubset(delete_labels):
        os.remove(json_path)
        print(f"Deleted entire file: {filename}")
        continue

    # Remove only the unwanted annotations
    new_shapes = [shape for shape in shapes if shape["label"] not in delete_labels]

    if len(new_shapes) != len(shapes):
        data["shapes"] = new_shapes
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Removed deleted labels from: {filename}")

#%%
# Label renaming map (don't care about Delete column here)
rename_map = dict(zip(df["LabelmeCode"], df["Kees code"]))

# Process JSON files
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    updated = False

    for shape in shapes:
        old_label = shape["label"]
        if old_label in rename_map:
            new_label = rename_map[old_label]
            if new_label != old_label:
                shape["label"] = new_label
                updated = True

    if updated:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Renamed labels in: {filename}")
