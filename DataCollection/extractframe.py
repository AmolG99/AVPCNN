import cv2
import os

def extractframe(cap, frame_count, frame_interval, output_dir):
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Couldn't read no frame")
    
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
    