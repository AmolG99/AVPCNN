# Video frame extractor & preprocessor using PIL and cv2
# A Michael Lance & Amol Gupta production
# 6/21/2024
# 6/24/2024
#-----------------------------------------------------------------------------------------------------------#
import os
import cv2
from PIL import Image

video_path = 'test_vid.MP4'

cap = cv2.VideoCapture(video_path)

output_dir = 'captured_frames'
os.makedirs(output_dir, exist_ok=True)

if not cap.isOpened():
    raise Exception("camera'nt")
frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Couldn't read no frame")
    
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   
        break
    
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()