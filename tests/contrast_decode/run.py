import vllm

if __name__ == "__main__":
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    positive_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    negative_model = "meta-llama/Meta-Llama-3-8B-Instruct"

    dummy_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
    worker = vllm.LLM(
        model=base_model,
        positive_model=positive_model,
        negative_model=negative_model,
        sampler_alpha=0.5,
    )

    outputs = worker.generate(dummy_prompts)
    for output in outputs:
        print(output.text)
