from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import time
import torch
app = Flask(__name__)

# Load model and processor (moved outside of the route for efficiency)
model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

@app.route('/describe_image', methods=['POST'])
def describe_image():
    # Get image URL from request
    image_url = request.json.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Prepare for inference
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

    # Inference
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    model.eval()
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

if __name__ == '__main__':
    app.run(debug=True)