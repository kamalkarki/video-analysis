# write a function which read config.json and set the environment variables and check if the model us not present in cache then download it
import json
import os
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

def print_current_working_directory():
    print("Current working directory:")
    print(os.getcwd())


def check_model_cache(model_id):
    filepath = try_to_load_from_cache(filename=model_id)
    if isinstance(filepath, str):
        # file exists and is cached
        return True
    elif filepath is _CACHED_NO_EXIST:
        # non-existence of file is cached
        return False
    else:
        return False

print_current_working_directory()

# bool = check_model_cache("Qwen2-VL-2B-Instruct-GPTQ-Int8")
# print(bool)

