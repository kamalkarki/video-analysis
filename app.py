from flask import Flask, request, jsonify
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import time
import torch
from functools import lru_cache
import os
import json
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

# from models.llama3.reference_impl.generation import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer_path = '/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct/tokenizer.model'
ckpt_dir = "/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct"

app = Flask(__name__)

# load the config file
config = json.load(open('config_folder/config.json'))
# print(config)
model_name = config["vision_model"]["qwen2"]["model_name"]
model_source = config["vision_model"]["qwen2"]["model_source"]

text_model_name = config["text_model"]["qwen2"]["model_name"]
image_folder = config["config"]["image_folder"]

# TODO:
# check if the model is already downloaded , I have tried to implement cache for hugging face models but it was not working
# here setting manually from the download path due to low memory in C drive

model_dir = snapshot_download(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir)

# Enable model evaluation mode
model.eval()


def process_image(image_url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda")


@app.route('/describe_video', methods=['POST'])
def describe_video():
    start_time = time.time()
    video_name = request.json.get('video_name')
    if not video_name:
        return jsonify({"error": "No video name provided"}, 400)

    video_path = os.path.join(config["config"]["video_folder"], video_name)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    inputs = process_vision_info(messages)

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    end_time = time.time()
    time_taken = end_time - start_time

    return jsonify({
        "description": output_text[0],
        "time_taken": time_taken
    })


@app.route('/describe_image_directory', methods=['POST'])
def describe_image_directory():
    image_directory = request.json.get('image_directory')
    if not image_directory:
        return jsonify({"error": "No image directory provided"})

    image_directory = os.path.join(config["config"]["image_folder"], image_directory)

    # read all the images from the directory
    images = []
    for file in os.listdir(image_directory):
        if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            images.append(os.path.join(image_directory, file))

    image_descriptions = []
    # Inference
    for image in images:
        # Call the internal /describe_image endpoint using Flask's test client
        with app.test_client() as client:
            payload = {
                "image_urls": image
            }
            headers = {"Content-Type": "application/json"}
            response = client.post('/describe_image', json=payload, headers=headers)
            image_description = response.get_json()
            image_descriptions.append(image_description)

    return jsonify({
        "image_descriptions": image_descriptions
    })


@app.route('/describe_image_base64', methods=['POST'])
def describe_image_base64():
    start_time = time.time()

    data_type = request.json.get('type')
    image_base64 = request.json.get('image')

    if not image_base64:
        return jsonify({"error": "No image base64 provided"})

    messages = [
        {
            "role": "user",
            "content": [
                {"type": data_type, "image": image_base64},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    end_time = time.time()
    time_taken = end_time - start_time

    return jsonify({
        "image_url": "test_base64",
        "description": output_text[0],
        "time_taken": time_taken
    })


@app.route('/summarize_llama', methods=['POST'])
def summarization():
    context = request.json.get('data')
    from models.llama3.reference_impl.generation import Llama
    tokenizer_path = '/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct/tokenizer.model'
    ckpt_dir = "/home/kamlesh/.llama/checkpoints/Llama3.2-3B-Instruct"

    def run_main(
            ckpt_dir: str = ckpt_dir,
            temperature: float = 0.3,
            top_p: float = 0.9,
            max_seq_len: int = 1654,
            max_batch_size: int = 4,
            max_gen_len: int = 412,
            model_parallel_size: Optional[int] = None,
    ):
        # print(ckpt_dir)
        # tokenizer_path = str(tokenizer_path)
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )

        prompt = f"""    
            You will be provided a question and context, please do the following steps.
            Understand the Question: Carefully read and comprehend the query before proceeding to the next step.
            Find concise answer from the provided context.
            Information about context: After going through a video of an expert has written a textual description for each each second of video,
            go through the "description" and give a summary of all the frames also analyze what happened ?
            Return only the generated answer, nothing else.
            question: {query}
            context: {context}
            """

        result = generator.text_completion(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=False,
        )

        result.generation
        data.append(json.dumps(response.json(), indent=2))
        return data

    return fire.Fire(run_main)


@app.route('/summarize_qwen', methods=['POST'])
def summarize():
    context = request.json.get('text')
    print("context", context)
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    question = 'Give a summary of all the frames also analyze what happened in the video?'
    # now the llm calls
    prompt_1 = f"""    
            You will be provided a question and context, please do the following steps.
            Understand the Question: Carefully read and comprehend the query before proceeding to the next step.
            Information about context: You are provided a text context which contains the description of video frames.
            Extracted for each second of the video by an analyst. Go through each frame description and try to understand what happened in the video.
            Find concise answer from the provided context.
            Return only the generated answer, nothing else.
            question: {question}
            context: {context}
            """
    prompt_2 = f"""    
            You are provided a context which contains two keys "frame" and "description".
            Description is the description of the video frame at a perticular second, generate a summary of all the frames also analyze what happened in the video.
            context: {context}
            """
    messages = [
        {"role": "system",
         "content": "You are a research assistant helping a user find an informative & summarized answer from context"},
        {"role": "user",
         "content": prompt_2},

    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # make a json object with key "summary" and value as the response
    response = {"summary": response}
    return jsonify(response)


@app.route('/describe_image', methods=['POST'])
def describe_image():
    image_url = request.json.get('image_urls')
    if not image_url:
        return jsonify({"error": "No image URL provided"})

    inputs = process_image(image_url)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    end_time = time.time()
    time_taken = end_time - start_time

    return jsonify({
        "image_url": image_url,
        "description": output_text[0],
        "time_taken": time_taken
    })


if __name__ == '__main__':
    app.run(debug=False, threaded=False, port=6000)