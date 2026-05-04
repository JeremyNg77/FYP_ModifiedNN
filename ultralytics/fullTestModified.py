from ultralytics import YOLO

def main():
    # 1. Load the BEST weights produced by your training script
    model = YOLO(r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\ultralytics\runs\detect\train9\weights\best.pt") 

    print("Initialising custom hyperparameter full testing...")
    
    # 2. Run validation with your custom hyperparameters
    metrics = model.val(
        data=r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\datasets\FVP-Baseline.v17-vfinal.yolov8\data.yaml",
        split="val",  
        
        # --- YOUR HYPERPARAMETERS ---
        conf=0.720,    
        iou=0.10,     
        
        # --- OUTPUT SETTINGS ---
        plots=True,   
        save_json=True 
    )

    # 3. Print Overall Metrics
    print("\n" + "="*45)
    print("OVERALL METRICS")
    print("="*45)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 45)
    print(f"{'Overall mAP@50':<25} | {metrics.box.map50:.4f}")
    print(f"{'Overall mAP@50-95':<25} | {metrics.box.map:.4f}")
    print(f"{'Overall Precision':<25} | {metrics.box.mp:.4f}")
    print(f"{'Overall Recall':<25} | {metrics.box.mr:.4f}")
    
    # Inference speed is stored in a dictionary (preprocess, inference, loss, postprocess)
    # We extract the 'inference' time per image in milliseconds
    print(f"{'Inference Speed':<25} | {metrics.speed['inference']:.1f} ms/img")

    # 4. Print Class-Specific Metrics
    print("\n" + "="*45)
    print("CLASS-SPECIFIC METRICS")
    print("="*45)
    print(f"{'Class':<12} | {'Metric':<15} | {'Value':<10}")
    print("-" * 45)
    
    # Extract the internal class indices evaluated during validation
    evaluated_classes = metrics.box.ap_class_index
    
    # Loop through each evaluated class to get its specific scores
    for i, class_idx in enumerate(evaluated_classes):
        # Map the index to the actual string name (e.g., "Part 1")
        class_name = model.names[class_idx]
        
        # --- THE FIX IS HERE ---
        # Extract the specific mAPs using the index 'i' from 'ap50' and 'ap'
        class_map50 = metrics.box.ap50[i]
        class_map_50_95 = metrics.box.ap[i]
        
        print(f"{class_name:<12} | {'mAP@50':<15} | {class_map50:.4f}")
        print(f"{'':<12} | {'mAP@50-95':<15} | {class_map_50_95:.4f}")
        print("-" * 45)

if __name__ == "__main__":
    main()