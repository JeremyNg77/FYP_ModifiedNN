import os

def process_labels_in_dir(labels_dir):
    """Scans a single directory and converts polygon labels to bounding boxes."""
    if not os.path.exists(labels_dir):
        print(f"⚠️ Warning: Directory not found: {labels_dir}")
        return 0

    print(f"🔧 Scanning dataset in '{labels_dir}' for Polygons...")
    fixed_files = 0

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
            
        label_path = os.path.join(labels_dir, label_file)
        new_lines = []
        file_needs_rewrite = False
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            
            cls_id = parts[0]
            values = [float(x) for x in parts]
            
            # If it's ALREADY a standard box (Class + cx, cy, w, h)
            if len(values) == 5:
                new_lines.append(line.strip())
            else:
                # It's a POLYGON! Let's convert it to a standard box.
                coords = values[1:]
                xs = coords[0::2]
                ys = coords[1::2]
                
                # Calculate bounding box dimensions
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                cx = min(xs) + (w / 2)
                cy = min(ys) + (h / 2)
                
                # Create the new standard YOLO line (5 values)
                new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                file_needs_rewrite = True
        
        # Rewrite the file to apply the conversions permanently
        if file_needs_rewrite:
            with open(label_path, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            fixed_files += 1

    return fixed_files


def main():
    # 1. Setup your base path
    base_dir = r"datasets/FVP-Baseline.v17-vfinal.yolov8"
    
    # 2. Define the subfolders you want to clean
    # (You can add "test/labels" here later if you have a test set)
    folders_to_clean = [
        "train/labels",
        "valid/labels"
    ]
    
    total_fixed_across_all_folders = 0

    # 3. Loop through each folder and run the conversion
    for folder in folders_to_clean:
        full_path = os.path.join(base_dir, folder)
        fixed_count = process_labels_in_dir(full_path)
        total_fixed_across_all_folders += fixed_count

    print("-" * 40)
    print(f"✅ Total Conversion Complete! Updated {total_fixed_across_all_folders} files across all folders.")

if __name__ == "__main__":
    main()