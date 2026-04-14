from ultralytics import YOLO

# You MUST wrap the execution in this block on Windows
if __name__ == '__main__':
    
    # 1. Load your fully trained model
    model = YOLO(r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\ultralytics\runs\detect\train6\weights\best.pt')

    # 2. Run the evaluation on the TEST set
    # This applies your strict NMS rules to the final exam
    print("Starting Final Test Set Evaluation...")
    metrics = model.val(
        data=r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\datasets\FVP-Baseline.v13-v13.yolov8\data.yaml', 
        split='val',         # Note: You have this set to 'val' to test the validation folder
        iou=0.10,            # Your aggressive overlap penalty
    )

    # 3. Print the core results for your report
    print("\n--- FINAL REPORT METRICS ---")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")