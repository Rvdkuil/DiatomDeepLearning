##################################
## Remove and group annotations ##
##################################

import os
import json

# Folder containing your annotation files
input_folder = "D:/sorted_images/combi"  

# Labels to remove
labels_to_remove = {"MMU"}  # <-- Add/remove labels as needed

# Loop through all JSON files
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    # Check which labels are present
    all_labels = {shape["label"] for shape in shapes}

    if all_labels.issubset(labels_to_remove):
        # All labels are ones to be removed → delete the file
        os.remove(json_path)
        print(f"Deleted: {filename}")
    else:
        # Keep only the shapes not in the labels_to_remove list
        new_shapes = [shape for shape in shapes if shape["label"] not in labels_to_remove]

        # Update and save the JSON if changes were made
        if len(new_shapes) != len(shapes):
            data["shapes"] = new_shapes
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated: {filename}")
