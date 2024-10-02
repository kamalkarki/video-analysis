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

app = Flask(__name__)
url = 'http://192.168.0.106:1234/v1/chat/completions'

@app.route('/summarize', methods=['POST'])
def describe_image():
    context = request.json.get('data')
    # now the llm calls   
    prompt_1 = f"""    
            Following is the description of a survillance video having a length of 30 seconds.
            Now for each second there is text description, go through the "description" and analyze what happend ?
            context: {context}
            """

    messages = [
        {
            "role": "system",
            "content": "You are an expert analyst who is provided with textual data. Your job is to summarize the data or to find informative answer from context provided."},
        
        {"role": "user",
         "content": prompt_1},

    ]

    # section modified to use json output from llm
    payload = json.dumps({
        "model": Config.ans_1_model_name,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": "true",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "answer"
                    ]
                }
            }
        },
        "temperature": 0.3,
        "max_tokens": -1,
        "Stream": False
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload).json() # this will convert it to dictionary # todo
    # Parse the content as JSON
    try:
        content = json.loads(response["choices"][0]["message"]["content"])
        answer_1 = content["answer"]
        print("step_1", answer_1)
    except KeyError:
        print("Exception in answer, could be null")
        answer_1 = None


# if __name__ == '__main__':
#     app.run(debug=False, threaded=False,port=6000)

# import torch
# from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
# outputs = pipe(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
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
print(response)
