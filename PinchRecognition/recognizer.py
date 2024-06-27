# Media pipe based gesture recognition script 
# A Michael Lance & Amol Gupta experience
# 6/27/2024
# ------------------------------------------------------------------------------------------------------------------------#
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# path to the pretrained model itself, courtesy of Google
model_path = "gesture_recognizer.task"

DESIRED_WIDTH = 480
DESIRED_HEIGHT = 480

# instance everything from media pipe that we will need to interact with the model
# Pulling methods from modules and passing arguments is so hot right now
base_options = python.BaseOptions(model_asset_path='PinchRecognition/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

image_file_names = []
images = []
results = []



for filename in os.listdir("DataCollection/extracted_frames"):
    image_file_names.append("DataCollection/extracted_frames/" + filename)

# print(image_file_names) # Ensure image paths are correctly ingested

for image_file_name in image_file_names:
    image = mp.Image.create_from_file(image_file_name)
    
    
    recognition_result = recognizer.recognize(image)

    # STEP 5: Process the result. In this case, visualize it.
    images.append(image)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
