import os
from ultralytics import YOLO

# 1. Check if paths exist first!
model_path = r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\ultralytics\runs\detect\train6\weights\best.pt'
image_dir = r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\datasets\FVP-Baseline.v14-v14.yolov8\valid\images'

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
if not os.path.exists(image_dir):
    print(f"ERROR: Image directory not found at {image_dir}")
else:
    print(f"Success: Found {len(os.listdir(image_dir))} files in the image directory.")

# 2. Load Model
print("Loading model... please wait.")
model = YOLO(model_path)

# 3. Run Prediction with 'stream=True' to prevent memory freezing
print("Starting prediction...")
results = model.predict(
    source=image_dir, 
    save=True, 
    conf=0.531, 
    verbose=True  # This forces the terminal to show every image it processes
)

print("Done! Check the 'runs/detect/' folder for a new 'predict' directory.")