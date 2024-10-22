import time
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    base_model = "Qwen/Qwen2.5-3B"
    positive_model = "Qwen/Qwen2.5-1.5B-Instruct"
    negative_model = "Qwen/Qwen2.5-1.5B"

    dummy_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
    worker = LLM(
        model=base_model,
        cd_positive_model=positive_model,
        cd_negative_model=negative_model,
        cd_decoding_alpha=1,
        gpu_memory_utilization=0.75,
    )

    sampling_params = SamplingParams(
        max_tokens=1024,
    )
    # time the generation
    start_time = time.time()
    outputs = worker.generate(dummy_prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    for output in outputs:
        print(output.outputs[0].text)
