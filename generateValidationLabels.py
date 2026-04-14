import cv2
import os
import glob

def main():
    # 1. Define your paths
    images_dir = r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\datasets\FVP-Baseline.v10-v10.yolov8\valid\images'
    labels_dir = r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\datasets\FVP-Baseline.v10-v10.yolov8\valid\labels'
    
    # Where to save the output images
    output_dir = r'C:\Users\efyjn8\Desktop\FYP_ModifiedNN-main\FYP-ModifiedNN\All_Ground_Truth_Images'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Define your Classes and Colors (BGR format for OpenCV)
    classes = ["Part 1", "Part 2", "Part 3", "Part 4", "Part 5", "Part 6"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    print(f"Starting generation... looking for images in {images_dir}")

    # 3. Loop through every image in the validation folder
    for img_path in glob.glob(os.path.join(images_dir, '*.jpg')):
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape

        # 4. Check if a label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0: continue # Skip empty lines
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # --- THE NEW FIX: Handle both Boxes and Polygons ---
                if len(coords) == 4:
                    # Standard format: x_center, y_center, width, height
                    x_center, y_center, width, height = coords
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                else:
                    # Segmentation format: x1, y1, x2, y2, x3, y3...
                    # Grab all X's and Y's, then find the min/max to create the tightest bounding box
                    xs = coords[0::2]  # Get all the X coordinates (even indices)
                    ys = coords[1::2]  # Get all the Y coordinates (odd indices)
                    x1 = int(min(xs) * w)
                    x2 = int(max(xs) * w)
                    y1 = int(min(ys) * h)
                    y2 = int(max(ys) * h)
                # --------------------------------------------------

                # Draw the bounding box and text label
                color = colors[class_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, classes[class_id], (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Save the final annotated image
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)

    print(f"Finished! Ground truth images saved to: {output_dir}")

if __name__ == '__main__':
    main()