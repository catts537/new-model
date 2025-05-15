from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import numpy as np
from collections import Counter, defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8n.pt')
vehicle_classes = [2, 3, 5, 7]

tracker = DeepSort(max_age=30)

video_path = 's.mp4'
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Vehicle Tracking Cleaned', cv2.WINDOW_NORMAL)

track_histories = {}

# Store vehicle IDs counted per direction to avoid multiple counting
counted_ids_per_direction = defaultdict(set)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_point = (frame_width // 2, frame_height // 2)

def compute_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))  # -180 to 180
    return angle

def angle_to_direction(angle):
    angle = (angle + 360) % 360
    if 337.5 <= angle or angle < 22.5:
        return "East"
    elif 22.5 <= angle < 67.5:
        return "South-East"
    elif 67.5 <= angle < 112.5:
        return "South"
    elif 112.5 <= angle < 157.5:
        return "South-West"
    elif 157.5 <= angle < 202.5:
        return "West"
    elif 202.5 <= angle < 247.5:
        return "North-West"
    elif 247.5 <= angle < 292.5:
        return "North"
    elif 292.5 <= angle < 337.5:
        return "North-East"
    else:
        return "Unknown"

# Initialize counts
intersection_counts = Counter()

directions = {
    "East": 0, "South-East": 45, "South": 90, "South-West": 135,
    "West": 180, "North-West": 225, "North": 270, "North-East": 315
}

line_length = 200

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device)[0]
    detections = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if conf.item() < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        if class_id in vehicle_classes:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), class_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id in track_histories:
            prev_cx, prev_cy = track_histories[track_id]
            angle = compute_angle((prev_cx, prev_cy), (cx, cy))

            if np.hypot(cx - prev_cx, cy - prev_cy) > 5:
                direction = angle_to_direction(angle)
                # Count only if this vehicle ID is not already counted for this direction
                if track_id not in counted_ids_per_direction[direction]:
                    intersection_counts[direction] += 1
                    counted_ids_per_direction[direction].add(track_id)

        track_histories[track_id] = (cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Draw reference direction lines
    for dir_label, angle in directions.items():
        rad = np.radians(angle)
        end_x = int(center_point[0] + line_length * np.cos(rad))
        end_y = int(center_point[1] + line_length * np.sin(rad))
        cv2.arrowedLine(frame, center_point, (end_x, end_y), (0, 255, 255), 2, tipLength=0.05)
        cv2.putText(frame, dir_label, (end_x, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display counts
    y_offset = 30
    for dir_label in directions.keys():
        count = intersection_counts.get(dir_label, 0)
        cv2.putText(frame, f"{dir_label}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 25

    cv2.imshow('Vehicle Tracking Cleaned', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n=== Final Intersection Counts ===")
for dir_label in directions.keys():
    print(f"{dir_label}: {intersection_counts.get(dir_label, 0)} vehicles")
