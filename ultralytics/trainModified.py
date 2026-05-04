from ultralytics import YOLO
import time  # <-- Added this to handle background threading

def main():
    # 1. Load a model
    # By using 'yolov8n.pt', the library will automatically download 
    # the weights for you if they aren't already in your folder.
    model = YOLO("yolov8n.pt") 

    # 2. Train the model
    print("Starting training... (Full batch plotting is ENABLED via source code mod)")
    results = model.train(
        data=r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\datasets\FVP-Baseline.v17-vfinal.yolov8\data.yaml", 
        epochs=150, 
        imgsz=640,
        patience=50,
        batch=8,
        device=0, # Change to device='cpu' if you don't have an NVIDIA GPU
        workers=2,
        amp=True,
        plots=True, # <-- Explicitly forces YOLO to generate evaluation graphs and collages
        
        # --- FYP DOMAIN-SPECIFIC OPTIMIZATIONS ---
        
        # 1. Geometry / Physics Constraints
        flipud=0.0,    # Heavy parts do not travel upside down on the conveyor
        degrees=0.0,   # Parts stay plumb on their hooks; no extreme rotation
        
        # 2. Lighting / Glare Robustness
        hsv_v=0.6,     # High brightness variance to mimic harsh factory lighting
        hsv_s=0.5,     # High saturation variance to handle metallic reflections
        
        # 3. Shape-IoU Synergy
        box=9.0        # Increased from default 7.5. Forces the model to strictly penalize 
                       # loose bounding boxes, keeping the tracking point dead center.
    )

    # 3. The Threading Trick (Crucial for your FYP modification)
    # Because you removed the limit, the final validation phase will try to save 
    # a collage for every single image. We pause the script to let OpenCV finish writing to the hard drive.
    print("Training finished! Waiting 15 seconds for background threads to save all validation collages...")
    time.sleep(15)
    print("Done! You can now check your 'runs/detect/train' folder for all the batch images.")

if __name__ == "__main__":
    main()