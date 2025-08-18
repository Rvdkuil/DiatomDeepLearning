#########################################################################
## Code to use YOLO with SAHI on a single image and export the visuals ##
#########################################################################

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

model_path="C:/Users/Gebruiker/runs/segment/train158/weights/last.pt"
image_path="D:/tests/SB23_45.5-46_Autofocus_color_63x_0.tif"
export_dir="E:/"
#%%
# Instantiate the model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.80,
    device=0
)

#%%
# Perform the inference
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=1216,
    slice_width=1216,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    postprocess_match_threshold=0.1,
    postprocess_class_agnostic = False,
    postprocess_type="NMS"
)

# Export visuals
result.export_visuals(export_dir=export_dir, text_size = 1, rect_th = 1)