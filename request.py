import requests
import json

def api_image_request():
    url = "http://127.0.0.1:6000/describe_image"
    payload = {
        "image_urls": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    print(response.json())

def api_image_request_2():
    url = "http://127.0.0.1:7000/summarize"
    payload = {
        "image_urls": "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    print(response.json())


def api_check_model_cache():
    url = "http://localhost:6000/check_model_cache"
    payload = {
        "model_id": "qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(response.json())


def check_model_cache():
    pass

#sudo kill $(sudo lsof -t -i:6000)
def api_video_request():
    url = "http://127.0.0.1:6000/describe_video"
    payload = {
        "video_name": "Driver_zooms_through_red_light_plows_into_two_cars.mp4"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(response.json())

def api_image_directory_request():
    url = "http://127.0.0.1:6000/describe_image_directory"
    payload = {
        "image_directory": "Driver_zooms_through_red_light_plows_into_two_cars"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print(response.json())


# api_image_request_2() #{'summary': 'The image depicts a serene beach scene at sunset with a person sitting on the sand, reading a book, and a dog sitting next to them, both facing the ocean. The sky is clear with a gradient of warm colors, and the water is calm.', 'time_taken': 142.61578011512756}
# api_image_request()
# {'description': 'The image depicts a serene beach scene with a woman and her dog. The woman is sitting on the sand, wearing a plaid shirt and black pants, and appears to be smiling as she high-fives the dog. The dog, which has a yellow coat and is wearing a colorful harness, is sitting next to her, also smiling and raising its front paw in a gesture of friendship or playfulness. The background shows the ocean with gentle waves, and the sky is clear with a soft glow from the setting or rising sun, casting a warm light over the entire scene. The overall atmosphere is peaceful and joyful, capturing a moment of connection', 'time_taken': 64.25123119354248}



# api_video_request()
api_image_directory_request()

