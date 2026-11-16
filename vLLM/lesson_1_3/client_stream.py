from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

stream = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[
        {
            "role": "user",
            "content": (
                "Explain how continuous batching improves GPU utilization "
                "in five simple sentences."
            ),
        }
    ],
    temperature=0,
    max_tokens=4096,
    stream=True,
    stream_options={
        "include_usage": True,
    },
    extra_headers={
        "X-Request-Id": "lesson13-sdk-stream-001",
    },
)

for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta

        if delta.content:
            print(
                delta.content,
                end="",
                flush=True,
            )

    if chunk.usage is not None:
        print()
        print()
        print("Usage:", chunk.usage)