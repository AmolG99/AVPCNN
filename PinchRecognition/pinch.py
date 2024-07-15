import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize MediaPipe Drawing utility
mp_drawing = mp.solutions.drawing_utils

def is_pinching(index_finger_tip, thumb_tip):
    # Calculate Euclidean distance between index finger tip and thumb tip
    distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5
    if distance < 0.01:  # Threshold for pinch gesture
        return True
    return False

# Directory containing extracted frames
input_dir = 'DataCollection/extracted_frames'

# Define the region of interest (ROI)
roi_top = 170
roi_bottom = roi_top + 2000
roi_left = 170
roi_right = roi_left + 2000

frames = os.listdir(input_dir)
frames.sort()

for frame_filename in frames:
    frame = cv2.imread(os.path.join(input_dir, frame_filename))
    #print(frame.shape)
    # Crop the frame to the ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    # Convert the BGR image to RGB before processing
    results = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:  # If any hand is detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks for index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Check if pinch gesture is made
            if is_pinching(index_finger_tip, thumb_tip):
                print(f"Pinch gesture detected in {frame_filename}!")
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Convert the RGB image back to BGR for display
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
            # Draw bounding box around the hand
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])), (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands', roi)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

hands.close()
cv2.destroyAllWindows()


# Read the CSV file
data = pd.read_csv('./DataCollection/ground_truth_DSCN0160.csv')

# Initialize counters for true positives, false positives, true negatives, false negatives
TP = FP = TN = FN = 0

# Iterate over the rows of the DataFrame
for _, row in data.iterrows():
    # Get the algorithm's output and the ground truth for this frame
    algorithm_output = row['algorithm_output']
    ground_truth = row['ground_truth']

    # Update counters
    if algorithm_output == 1 and ground_truth == 1:
        TP += 1
    elif algorithm_output == 1 and ground_truth == 0:
        FP += 1
    elif algorithm_output == 0 and ground_truth == 1:
        FN += 1
    elif algorithm_output == 0 and ground_truth == 0:
        TN += 1

# Calculate and print sensitivity, specificity, and accuracy
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Sensitivity: {sensitivity * 100}%")
print(f"Specificity: {specificity * 100}%")
print(f"Accuracy: {accuracy * 100}%")

# Print the counts of true positives, true negatives, false positives, and false negatives
print(f"True Positives: {TP}")
print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")


# Pinch Group Recognition Algorithm

# take recognized pinch and construct an array containing that pinch and n number frames to either index location, write that to a file to read in head tracking.

