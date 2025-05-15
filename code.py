import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8n.pt')
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
tracker = DeepSort(max_age=30)

video_path = 's.mp4'
cap = cv2.VideoCapture(video_path)

# Globals for drawing lines
lines = []  # each line: ((x1, y1), (x2, y2), direction_label)
drawing = False
current_line = []

def get_line_direction(p1, p2):
    """Returns 'Left_to_Right', 'Right_to_Left', 'Top_to_Bottom', 'Bottom_to_Top' based on line orientation"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    if 45 <= angle < 135:
        return "Top_to_Bottom"
    elif 225 <= angle < 315:
        return "Bottom_to_Top"
    elif 135 <= angle < 225:
        return "Right_to_Left"
    else:
        return "Left_to_Right"

def mouse_callback(event, x, y, flags, param):
    global drawing, current_line, lines, first_frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            current_line = [(x, y)]
        elif drawing and len(current_line) == 1:
            current_line.append((x, y))
            # Add line with direction label
            direction = get_line_direction(current_line[0], current_line[1])
            lines.append((current_line[0], current_line[1], direction))
            print(f"Line drawn: {current_line[0]} to {current_line[1]} Direction: {direction}")
            drawing = False
            current_line = []
            # Redraw lines on first frame display
            draw_lines(first_frame_display)

def draw_lines(frame):
    for pt1, pt2, direction in lines:
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        # Put direction label near middle of line
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(frame, direction, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def check_line_crossing(p1, p2, line_p1, line_p2):
    """Check if segment p1->p2 crosses line line_p1->line_p2 using vector cross product sign change"""
    def side(a, b, c):
        # returns signed area of triangle a,b,c * 2
        return (c[0] - a[0])*(b[1] - a[1]) - (c[1] - a[1])*(b[0] - a[0])
    side1 = side(line_p1, line_p2, p1)
    side2 = side(line_p1, line_p2, p2)
    return side1 * side2 < 0  # crossing if signs differ

# ===

print("Draw counting lines on the first frame by clicking two points per line.")
print("Press 's' to start processing video after drawing lines.")

# Read first frame to draw lines on
ret, first_frame_display = cap.read()
if not ret:
    print("Failed to read video")
    exit()

cv2.namedWindow('Draw Lines')
cv2.setMouseCallback('Draw Lines', mouse_callback)

while True:
    disp_frame = first_frame_display.copy()
    # Draw current lines and line being drawn
    draw_lines(disp_frame)
    if drawing and len(current_line) == 1:
        cv2.circle(disp_frame, current_line[0], 5, (0, 0, 255), -1)
    cv2.imshow('Draw Lines', disp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(lines) == 0:
            print("Draw at least one line!")
            continue
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow('Draw Lines')

# Now process video with lines and counting

track_histories = {}
counted_ids_per_line = [set() for _ in range(len(lines))]  # prevent multiple counting per line
counts_per_line = [0] * len(lines)

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
            prev_pos = track_histories[track_id]
            curr_pos = (cx, cy)

            # Check crossing for each line
            for i, (lp1, lp2, direction) in enumerate(lines):
                if track_id not in counted_ids_per_line[i]:
                    crossed = check_line_crossing(prev_pos, curr_pos, lp1, lp2)
                    if crossed:
                        counts_per_line[i] += 1
                        counted_ids_per_line[i].add(track_id)
                        print(f"Vehicle {track_id} crossed line {i} ({direction}). Total count: {counts_per_line[i]}")

        track_histories[track_id] = (cx, cy)

        # Draw bbox + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Draw counting lines and counts
    for i, (lp1, lp2, direction) in enumerate(lines):
        cv2.line(frame, lp1, lp2, (0, 255, 255), 3)
        mid = ((lp1[0] + lp2[0]) // 2, (lp1[1] + lp2[1]) // 2)
        cv2.putText(frame, f"{direction}: {counts_per_line[i]}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Vehicle Tracking with Counting Lines', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
