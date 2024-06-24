import cv2
import os

def extractframe(cap, frame_count):
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Couldn't read no frame")
    
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_filename, frame)
    