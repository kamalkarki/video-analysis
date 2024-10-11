# this is a script to extract frames from a video 
# as a pre-processing step for the video summarization task
# video file path is provided as an argument when running the script
# output frames are saved to a folder named 'video_frames' in the same directory as the video file

import cv2
import os
import argparse
import datetime
import json
import base64
import requests
# read frame extraction config
with open('config_folder/frame_extraction.json', 'r') as config_file:
    config = json.load(config_file)

OUTPUT_FOLDER = config['config']['output_folder']
INPUT_FOLDER = config['config']['input_folder']




def extract_frames(video_path):
    """
    Extract frames from a video file
    """
    print(f"Extracting frames from {video_path}")
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # get the duration of the video
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    print(f"Duration: {duration} seconds")

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate/ frame per second: {frame_rate}")
    # Calculate the interval between frames to extract

    extract_interval = int(frame_rate) # extract one frame per second
    print(f"Extract interval: {extract_interval}")
    
    
    # Create a timestamp for the file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a folder with the name of the video file
    if config['config']['time_stamp']:
        output_folder = os.path.join(video_name, timestamp)
    else:
        output_folder = video_name
    # create the output folder inside the main output folder
    output_folder = os.path.join(OUTPUT_FOLDER, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
   
    # Get the frame rate of the video   
    frame_index = 0
    while True:
        ret, frame = cap.read() # ret is a boolean variable that returns true if the frame is available. 
                                #   frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
        if not ret:
            break
        else:
            frame_index += 1
            if frame_index % extract_interval == 0:
                # Save the frame to the video_frames folder
                print(f"Extracting frame {frame_index}")
                frame_path = os.path.join(output_folder, f'frame_{frame_index}.jpg')
                print(f"Frame path: {frame_path}")
                cv2.imwrite(frame_path, frame)
    
    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_index + 1} frames from the video.")

def main():
    # parser = argparse.ArgumentParser(description="Extract frames from a video.")
    # parser.add_argument("video_path", type=str, help="Path to the video file")
    # args = parser.parse_args()
    video_name = "screen_recording.mp4"
    video_path = os.path.join(INPUT_FOLDER, video_name)
    # video_path = 'data/video/f1_car_vs_van.mp4'
    extract_frames(video_path)

if __name__ == "__main__":
    main()
