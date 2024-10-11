import base64
import gradio as gr
import requests
import json
def process_video(video_path):
    # Assuming the API endpoint for video processing is '/process_video'
    # and it's running on localhost:6000
    url = "http://localhost:6000/process_video"
    
    files = {'video': open(video_path, 'rb')}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return json.dumps(response.json(), indent=2)
    else:
        return f"Error: {response.status_code} - {response.text}"


def process_image(image_path):
    # Assuming the API endpoint for image processing is '/describe_image'
    # and it's running on localhost:6000
    url = "http://localhost:6000/describe_image"
    
    # Read the image file and encode it as base64
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare the payload with the base64 encoded image
    payload = {
        "type": "image",
        "image": f"data:image/jpeg;base64,{encoded_image}"
    }
    headers = {"Content-Type": "application/json"}

    # Send the request to the API
    response = requests.post(url, json=payload, headers=headers)
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return json.dumps(response.json(), indent=2)
    else:
        return f"Error: {response.status_code} - {response.text}"

# Update Gradio interface to include both video and image processing
iface = gr.Interface(
    fn= process_image,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Image(label="Upload Image")
    ],
    outputs=[
        gr.JSON(label="Video Processing Result"),
        gr.JSON(label="Image Processing Result")
    ],
    title="Video and Image Processing App",
    description="Upload a video or an image to process it using the local API server."
)

# Launch the app
iface.launch(server_name="0.0.0.0", server_port=7860)







