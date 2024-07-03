import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing utility
mp_drawing = mp.solutions.drawing_utils

def is_pinching(index_finger_tip, thumb_tip):
    # Calculate Euclidean distance between index finger tip and thumb tip
    distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5
    if distance < 0.05:  # Threshold for pinch gesture
        return True
    return False

# Directory containing extracted frames
input_dir = 'DataCollection/extracted_frames'

for frame_filename in os.listdir(input_dir):
    frame = cv2.imread(os.path.join(input_dir, frame_filename))

    # Convert the BGR image to RGB before processing
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

hands.close()
cv2.destroyAllWindows()
