import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# Check GPU availability for the Asus TUF F15
if torch.cuda.is_available():
    print(f"[INFO] Booting up using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[WARNING] CUDA not detected. Running on CPU.")

MODEL_PATH = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\ultralytics\FINAL MODEL.pt"
OUTPUT_PATH = "live_demo_recording.mp4" # Keeps a recording of your live demo!

# Camera Settings
# 0 is usually the built-in Asus webcam, 1 is the external USB camera
CAMERA_INDEX = 1 

# Line Coordinates (x1, y1) to (x2, y2)
# BE READY TO CHANGE THESE ON-SITE based on how you mount the camera!
LINE_START = (600, 0)
LINE_END = (600, 1080)

print("[INFO] Loading Model...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# ==========================================
# 2. INITIALIZATION
# ==========================================

# --- OPTION A: LIVE USB CAMERA (Uncomment these two lines for the Panasonic Demo) ---
print("[INFO] Connecting to USB Camera...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


# --- OPTION B: TEST VIDEO FILE (Currently active for your dry run) ---
#print("[INFO] Loading Test Video...")
#VIDEO_PATH = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\Test Video.mp4"
#cap = cv2.VideoCapture(VIDEO_PATH)


# Force the 2MP (1920x1080) resolution (Works for both video and live camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify what resolution actually opened
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30 # Fallback if camera doesn't report FPS
print(f"[INFO] Source initialized at {w}x{h} @ {fps} FPS")

# Setup Video Writer to record your demo
video_writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

track_history = {}
counts = {name: 0 for name in CLASS_NAMES.values()}

# Create a resizable window for the demo presentation
cv2.namedWindow("Panasonic Live Demo - YOLOv8", cv2.WINDOW_NORMAL)

def check_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

print("[INFO] Starting real-time tracking and counting... Press 'q' to quit.")

# Variables for FPS calculation
prev_time = 0

# ==========================================
# 3. VIDEO PROCESSING LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame. Is the camera unplugged?")
        break
        
    # Calculate Live FPS
    current_time = time.time()
    live_fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
        
    # Run YOLO with integrated tracking
    results = model.track(frame, persist=True, show=False, conf=0.5)[0]
    
    annotated_frame = frame.copy()
    
    # Draw the counting line
    cv2.line(annotated_frame, LINE_START, LINE_END, (0, 255, 255), 3)

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        class_ids = results.boxes.cls.int().cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy() 

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_center = (center_x, center_y)
            class_name = CLASS_NAMES[class_id]

            # 1. Draw the bounding box and center dot
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, current_center, 4, (0, 0, 255), -1)

            # 2. Label text
            label = f"{class_name} {conf:.2f}"
            
            # 3. Text Background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # 4. Draw Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- COUNTING LOGIC ---
            if track_id in track_history:
                previous_center = track_history[track_id]['prev_center']
                
                # Check intersection
                if check_intersection(LINE_START, LINE_END, previous_center, current_center):
                    if not track_history[track_id].get('counted', False):
                        counts[class_name] += 1
                        track_history[track_id]['counted'] = True
                        
                        # Change line color briefly to RED to visually confirm a count
                        cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 5)

            # Update history
            track_history[track_id] = {
                'class_name': class_name,
                'prev_center': current_center,
                'counted': track_history.get(track_id, {}).get('counted', False)
            }

    # --- DISPLAY METRICS ---
    # Draw FPS in Top Right Corner
    fps_text = f"FPS: {live_fps:.1f}"
    cv2.putText(annotated_frame, fps_text, (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Draw the Counts in Top Left Corner
    y_offset = 40
    for class_name, count in counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(annotated_frame, text, (22, y_offset + 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(annotated_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40

    # Save to video
    video_writer.write(annotated_frame)

    # Show live feed
    cv2.imshow("Panasonic Live Demo - YOLOv8", annotated_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('c'): # Press 'c' to clear counts (useful during calibration!)
        print("[INFO] Counts cleared manually.")
        counts = {name: 0 for name in CLASS_NAMES.values()}
        track_history.clear()

# ==========================================
# 4. CLEANUP & EXPORT
# ==========================================
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\n" + "="*30)
print("     FINAL INVENTORY COUNT")
print("="*30)
for name, count in counts.items():
    if count > 0:
        print(f" > {name}: {count}")
print("="*30)
print(f"[INFO] Demo recording saved to {OUTPUT_PATH}")