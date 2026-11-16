from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-0.6B"
HF_TOKEN=os.getenv("HF_TOKEN")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, hf_token=HF_TOKEN)

print("Vocab size: ", len(tokenizer))

# Test simple encoding
samples = [
    "Hello world!",
    "vLLM is fast.",
    "Xin chào, tôi đang học vLLM.",
]

for s in samples:
  encoded=tokenizer(s, add_special_tokens=False)
  print("------------------")
  print(f"Before encode: {s}")
  print(f"After encode: {encoded['input_ids']}")
  print(f"Len of token: {len(encoded['input_ids'])}")
  print(f"Decoded: {tokenizer.decode(encoded['input_ids'])}")

# Test chat template
messages=[{
    "role": "user",
    "content": "My name is Tu"
}]

chat_text=tokenizer.apply_chat_template(messages, add_generation_prompt=True, add_tokenize=False)
chat_ids=tokenizer.apply_chat_template(messages, add_generation_prompt=True, add_tokenize=True)

raw_text=messages[0].get("content")
raw_ids=tokenizer(raw_text, add_special_tokens=False)["input_ids"]

print("=" * 60)
print("Raw user text:")
print(raw_text)
print("Raw token count:", len(raw_ids))

print("=" * 60)
print("Formatted chat prompt:")
print(chat_text)

print("=" * 60)
print("Chat prompt token count:", len(chat_ids['input_ids']))
print("Chat template overhead:", len(chat_ids['input_ids']) - len(raw_ids))