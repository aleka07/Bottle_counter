import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8x.pt')

# Mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create RGB window and set mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video capture
cap = cv2.VideoCapture('test_bottle.mp4')

# Read COCO class list
with open("coco.txt", "r") as file:
    data = file.read()
    class_list = data.split("\n")

# Initialize variables
count = 0
tracker = Tracker()
cx1 = 350  # Changed from cy1 to cx1 for x-axis tracking
offset = 6
counter = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Run object detection
    results = model.predict(frame)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")

    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        if d == 39:
            list.append([x1, y1, x2, y2])

    # Update object tracker
    bbox_id = tracker.update(list)

    # Draw bounding boxes and track objects
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        if cx1 > (cx - offset) and cx1 < (cx + offset):  # Changed from y4 to x4 for x-axis tracking
            if counter.count(id) == 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 200), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                counter.append(id)

    # Draw tracking line
    cv2.line(frame, (cx1, 258), (cx1, 720), (0, 255, 0), 2)  # Changed from cy1 to cx1 for x-axis tracking

    # Display bottle count
    l = len(counter)
    cv2.putText(frame, f'Bottles: {l}', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("RGB", frame)

    # Break loop on ESC key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Write total count to a text file
with open("count.txt", "w") as file:
    file.write(f"The total count of bottles is {len(counter)}.\n")

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
