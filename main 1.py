import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Initialize the YOLO model
model = YOLO("yolov8s.pt")

# Line positions for entry and exit detection
cy1 = 200  # Entry line
cy2 = 300  # Exit line
offset = 15  # Margin of error for crossing lines

# Initialize counters and person states
counter_in = 0
counter_out = 0
person_states = {}

def get_video_source():
    """
    Prompt user to choose between a laptop webcam or CCTV feed.
    """
    print("Choose video source:")
    print("1. Laptop Webcam")
    print("2. CCTV Feed (RTSP URL)")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        return 0  # Webcam (live video)
    elif choice == "2":
        rtsp_url = input("Enter the RTSP URL for CCTV feed: ").strip()
        return rtsp_url  # RTSP URL for live CCTV
    else:
        print("Invalid choice. Defaulting to webcam.")
        return 0

# Get the live video source
video_source = get_video_source()
cap = cv2.VideoCapture('vidp.mp4')

# Ensure the video source is live
if not cap.isOpened():
    print("Error: Unable to access the video source.")
    exit()

# Initialize the tracker
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from the live video source.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))

    # Detect objects using YOLO
    results = model.predict(frame)
    detections = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, class_id = map(int, box[:6])
            if model.names[class_id] == "person":
                detections.append([x1, y1, x2, y2])

    # Update tracker with detections
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'ID {obj_id}', (x1, y1 - 10), scale=1, thickness=1)

        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Initialize person state if not present
        if obj_id not in person_states:
            person_states[obj_id] = {"crossed_in": False, "crossed_out": False}

        # Check if person crosses entry line
        if not person_states[obj_id]["crossed_in"] and cy1 - offset < cy < cy1 + offset:
            counter_in += 1
            person_states[obj_id]["crossed_in"] = True
            print(f"Person ID {obj_id} entered. Total In: {counter_in}")

        # Check if person crosses exit line
        if not person_states[obj_id]["crossed_out"] and cy2 - offset < cy < cy2 + offset:
            counter_out += 1
            person_states[obj_id]["crossed_out"] = True
            print(f"Person ID {obj_id} exited. Total Out: {counter_out}")

    # Draw lines for entry and exit
    cv2.line(frame, (0, cy1), (1020, cy1), (0, 255, 0), 2)
    cv2.line(frame, (0, cy2), (1020, cy2), (0, 0, 255), 2)

    # Display counts on the frame
    cvzone.putTextRect(frame, f'In: {counter_in}', (50, 50), scale=2, thickness=2)
    cvzone.putTextRect(frame, f'Out: {counter_out}', (50, 150), scale=2, thickness=2)

    # Show the live video feed with annotations
    cv2.imshow("Live Detection", frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
