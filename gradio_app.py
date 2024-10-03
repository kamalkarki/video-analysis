import gradio as gr
import cv2
from PIL import Image
import base64
import requests
import json
import ast 
def process_video(video):
    """
    Process video file
    """
    cap = cv2.VideoCapture(video)
    # get the duration of the video
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    extract_interval = int(frame_rate) * 1 # extract one frame per second 
    
    # Please increase the extract_interval if you want to extract less frames for faster processing
    # extract_interval = int(frame_rate) * 15
    
    # # Create a timestamp for the file name
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # # Get the video file name without extension
    # video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # # Create a folder with the name of the video file
    # if config['config']['time_stamp']:
    #     output_folder = os.path.join(video_name, timestamp)
    # else:
    #     output_folder = video_name
    # # create the output folder inside the main output folder
    # output_folder = os.path.join(OUTPUT_FOLDER, output_folder)
    # os.makedirs(output_folder, exist_ok=True)
    
    data = []
    content = {}
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
                # Convert captured image to JPG
                retval, buffer = cv2.imencode('.jpg', frame)

                # Convert to base64 encoding and show start of data
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                # encoded_image = base64.b64encode(frame).decode('utf-8')
                # Prepare payload
                payload = {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{encoded_image}"
                }
                
                # Send request to API
                url = "http://localhost:6000/describe_image_base64"
                headers = {"Content-Type": "application/json"}
                #print("payload: ")
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:

                    # extract the description from the response
                    description = response.json()["description"]

                    # remove the double quotes from the description
                    description = description.replace('"', '')

                    # add the description to the content dictionary with the frame index as the key
                    content[frame_index] = description

                    # now make the data such a way that it can be sent to the summarization call
                    
                else:
                    data.append(f"Error: {response.status_code} - {response.text}")
                    return data
    
    cap.release()
    
    payload = {
                    "type": "text",
                    "text": content
                }
    
    # now make the summarization call
    url = "http://localhost:6000/summarize_qwen"
    headers = {"Content-Type": "application/json"}
    print(payload)
    response = requests.post(url, json=payload, headers=headers)
    print("response in gradio: ", response)        
    print("response: ", response.json())        
    if response.status_code == 200:
        data.append(json.dumps(response.json(), indent=2))
    else:
        data.append(f"Error: {response.status_code} - {response.text}")



    return data

# Function to process media (image or video)
def process_media(media):
    if isinstance(media, str) and media.endswith(('.mp4', '.avi', '.mov')):
        # Video file
        data = process_video(media)
        
    elif isinstance(media, Image.Image):
        # Image file
        return media
    else:
        return "Unsupported file type."

    return data

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Upload an Image or Video")
    
    media_input = gr.File(label="Upload Image or Video", file_types=['image', 'video'])
    # output_display = gr.Image(label="Output Display"
    output_display=gr.JSON(label="Image Processing Result")

    media_input.change(process_media, inputs=media_input, outputs=output_display)

# Launch the app
demo.launch(server_name="0.0.0.0", server_port=7860)
