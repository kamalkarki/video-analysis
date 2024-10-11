#https://huggingface.co/microsoft/Phi-3.5-vision-instruct
from flask import Flask, request, jsonify
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
import io
from util import check_model_cache
import torch
torch.backends.cudnn.benchmark = True
app = Flask(__name__)
model_id = "microsoft/Phi-3.5-vision-instruct" 
import time


# ... existing model and processor initialization code ...
# Note: set _attn_implementation='eager' if you don't have flash_attn 
# NOTE: flash attention is installed but not working why????
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='eager'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 


@app.route('/summarize', methods=['POST'])
def summarize_slides():
    start_time = time.time()
    data = request.json
    urls = data.get('image_urls', [])
    url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    images = []
    placeholder = ""


    image_data = requests.get(url, stream=True).content
    images.append(Image.open(io.BytesIO(image_data)))
    placeholder += f"<|image_1|>\n"

    messages = [
        {"role": "user", "content": placeholder + "Describe this image."},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )
    # with torch.cuda.amp.autocast():
    #     inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
    #     generate_ids = model.generate(**inputs,
    #         eos_token_id=processor.tokenizer.eos_token_id,
    #         **generation_args
    #     )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    end_time = time.time()
    time_taken = end_time - start_time
    return jsonify({"summary": response,"time_taken":time_taken})

if __name__ == '__main__':
    app.run(debug=True, port=7000)