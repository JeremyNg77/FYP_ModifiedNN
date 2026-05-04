from ultralytics import YOLO

def main():
    # 1. Load your best trained model
    model = YOLO(r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\ultralytics\runs\detect\train2\weights\best.pt')

    # 2. Run PREDICT directly on the folder containing your validation images
    model.predict(
        # UPDATE THIS PATH to point directly to your validation images folder
        source=r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\datasets\FVP-Baseline.v10-v10.yolov8\valid\images', 
        project='FYP-ModifiedNN',        # Main folder name
        name='All_Validation_Images',    # Sub-folder name
        save=True,                       # Saves EVERY individual image with boxes drawn
        save_txt=True,                   # Saves the coordinates for each image
        conf=0.25,                       # Confidence threshold
        iou=0.6                          # NMS IoU threshold
    )

if __name__ == '__main__':
    main()