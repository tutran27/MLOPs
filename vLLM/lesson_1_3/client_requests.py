import requests


URL = "http://127.0.0.1:8000/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "X-Request-Id": "lesson13-python-requests-001",
}

payload = {
    "model": "qwen3.5-4b",
    "messages": [
        {
            "role": "user",
            "content": "Xin chao",
        }
    ],
    "temperature": 0,
    "chat_template_kwargs": {
        "enable_thinking": False
    },
    "max_tokens": 1024,
}

response = requests.post(
    URL,
    headers=headers,
    json=payload,
    timeout=120,
)

print("HTTP status:", response.status_code)
print("X-Request-Id:", response.headers.get("x-request-id"))

response.raise_for_status()

data = response.json()

print()
print("Assistant:")
print(data["choices"][0]["message"]["content"])

print()
print("Usage:")
print(data.get("usage"))