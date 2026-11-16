from vllm import LLM, SamplingParams

MODEL_PATH="Qwen/Qwen3-0.6B"

llm=LLM(model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.7, 
        max_num_seqs=4,
        )
        
sampling_params=SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

print(f"Loading model {MODEL_PATH}...")

messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Descript the cat in Vietnamese"}
]

result=llm.chat(messages=messages, 
                sampling_params=sampling_params,
                use_tqdm=True)

output = result
completion=output[0].outputs[0]

print("=" * 60)
print("ORIGINAL: ")
print(output)
print("=" * 60)
print(completion.text)


print()
print("=" * 60)
print("TOKEN INFORMATION")
print("=" * 60)
print("Prompt tokens:", len(output.prompt_token_ids))
print("Output tokens:", len(completion.token_ids))
print("Finish reason:", completion.finish_reason)