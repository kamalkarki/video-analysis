#https://github.com/vllm-project/vllm/discussions/8084
from huggingface_hub import parse_safetensors_file_metadata, get_safetensors_metadata

metadata = get_safetensors_metadata(repo_id="microsoft/Phi-3.5-mini-instruct")

def estimate_gpu_memory_for_weights(*, repo_id: str, revision: str = None):
    metadata = get_safetensors_metadata(repo_id=repo_id, revision=revision)
    parameter_count = metadata.parameter_count
    precision_size = {
        'FP32': 4,
        'FP16': 2,
        'BF16': 2,
        'INT8': 1,
        'INT4': 0.5
    }
    total_bytes = 0
    for precision, num_params in parameter_count.items():
        total_bytes += num_params * precision_size.get(precision, 0)
    return total_bytes / (1024 ** 3)
  
# print(estimate_gpu_memory_for_weights(repo_id="microsoft/Phi-3.5-mini-instruct"))
print(estimate_gpu_memory_for_weights(repo_id="microsoft/Phi-3.5-vision-instruct", revision="main"))