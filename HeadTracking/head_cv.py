import cv2
import numpy as np
from collections import deque, Counter

# Create a CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Open video file or a capturing device or a IP video stream for video capturing
cap = cv2.VideoCapture('DataCollection/DSCN0160.MP4')

# Read the first frame
ret, frame = cap.read()

# Select the bounding box of the VR headset in the first frame
bbox = cv2.selectROI("Tracking",frame, False)
tracker.init(frame, bbox)

# Define the number pad layout
num_pad = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [-1, 0, -1]  # -1 represents no key
])

# Get the size of the frame
height, width = frame.shape[:2]

# Initialize the sliding window and its size
window = deque(maxlen=10)
#yo
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    # Draw bounding box
    if success:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255,0,0), 3, 1)

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Map the center of the bounding box to the number pad
        key_x = int(center_x / width * 3)
        key_y = int(center_y / height * 4)
        key = num_pad[key_y, key_x]

        # Add the key to the sliding window
        window.append(key)

        # Calculate the top 5 keys and their probabilities
        counter = Counter(window)
        total = len(window)
        top_5_keys = counter.most_common(5)
        top_5_probs = [(key, count / total) for key, count in top_5_keys]

        # Print the top 5 keys and their probabilities
        print(f"Top 5 Keys: {top_5_probs}")

    else:
        cv2.putText(frame, "Lost", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xff == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
