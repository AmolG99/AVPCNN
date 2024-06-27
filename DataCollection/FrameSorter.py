# Video frame extractor & preprocessor using PIL and cv2
# # A Michael Lance & Amol Gupta production
# 6/21/2024
# 6/25/2024
#-----------------------------------------------------------------------------------------------------------#
import cv2
import os
from multiprocessing import Pool, cpu_count
import time


# Path to the video file
video_path = 'DataCollection/test_vid.MP4'

# Create a directory to save frames
output_dir = 'DataCollection/extracted_frames'
os.makedirs(output_dir, exist_ok=True)

def extract_frames(start_frame, end_frame, video_path, output_dir, interval, total_frames):
    #extract frame from video, save it as a flat file and open it 
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
            cv2.imwrite(frame_filename, frame)
    
        
        print(f" Extracted: {frame_num} / {total_frames}")
        
        

    cap.release()
    
def resize_image(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

def main():
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Total number of frames: {total_frames}")

    num_processes = cpu_count()
    chunk_size = total_frames // num_processes
    interval = 5  # Extract every third frame

    pool = Pool(processes=num_processes)
    tasks = []

    for i in range(num_processes):
        start_frame = i * chunk_size
        end_frame = (i + 1) * chunk_size if i != num_processes - 1 else total_frames
        tasks.append((start_frame, end_frame, video_path, output_dir, interval, total_frames))

    pool.starmap(extract_frames, tasks)
    pool.close()
    pool.join()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    prolapsed_time = end_time - start_time
    print(prolapsed_time)
