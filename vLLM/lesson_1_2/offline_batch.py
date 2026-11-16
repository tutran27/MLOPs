from vllm import LLM, SamplingParams

MODEL_PATH="Qwen/Qwen3-0.6B"

llm=LLM(model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.7, 
        max_num_seqs=4,
        )
        
sampling_params=SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

print(f"Loading model {MODEL_PATH}...")

conversations=[
    [
        {
            "role": "user",
            "content": "Explain prefill in one sentence.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Explain decode in one sentence.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Explain KV cache in one sentence.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Explain continuous batching in one sentence.",
        }
    ],
]

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    use_tqdm=True,
)


for i, output in enumerate(outputs):
    completion = output.outputs[0]

    print()
    print("=" * 60)
    print(f"REQUEST {i}")
    print("=" * 60)

    print("Prompt tokens:", len(output.prompt_token_ids))
    print("Output tokens:", len(completion.token_ids))
    print("Finish reason:", completion.finish_reason)
    print("Text:")
    print(completion.text)