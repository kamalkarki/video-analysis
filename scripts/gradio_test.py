import gradio as gr

import base64
import requests
import json

def process_image(image):
    if image is None:
        return "No image uploaded"
    # image is a path to the image
    # D:\research_v2\project\analysis\data\output\video_frames\Driver_zooms_through_red_light_plows_into_two_cars\frame_000000.jpg
    if image.startswith("D:\\"):
        image_name = image.split("\\")[-1]
    else:
        image_name = image.split("/")[-1]
    
    dir_path = "/mnt/d/research_v2/project/analysis/data/output/video_frames/Driver_zooms_through_red_light_plows_into_two_cars"
    # replace image extension to jpg
    image_path = dir_path + "/" + image_name.split(".")[0] + ".jpg"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare payload
    payload = {
        "type": "image",
        "image": f"data:image/jpeg;base64,{encoded_image}"
    }
    
    # Send request to API
    url = "http://localhost:6000/describe_image_base64"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return json.dumps(response.json(), indent=2)
    else:
        return f"Error: {response.status_code} - {response.text}"

# Create Gradio interface
# iface = gr.Interface(
#     fn=process_image,
#     inputs=gr.Image(type="filepath", label="Upload Image"),
#     outputs=gr.JSON(label="Image Processing Result"),
#     title="Image Processing App",
#     description="Upload an image to process it using the local API server."
# )


def process_video(video):
    if video is None:
        return "No video uploaded"
    
    # Extract video name from the path
    if video.startswith("D:\\"):
        video_name = video.split("\\")[-1]
    else:
        video_name = video.split("/")[-1]
    
    # send video name to the api
    url = "http://127.0.0.1:6000/describe_video"
    payload = {
        "video_name": "Driver_zooms_through_red_light_plows_into_two_cars.mp4"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Update Gradio interface to include video input
iface = gr.Interface(
    fn=process_video,
    inputs=gr.File(label="Upload Video", file_types=['image','video']),
    outputs=gr.JSON(label="Processing Result"),
    title="Image and Video Processing App",
    description="Upload an image or video to process it."
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)
