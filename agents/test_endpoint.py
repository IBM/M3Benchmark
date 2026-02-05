import httpx
import os

base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b"
headers = {"RITS_API_KEY": os.environ['RITS_API_KEY']}

# Test models endpoint
with httpx.Client() as client:
    resp = client.get(f"{base_url}/v1/models", headers=headers, timeout=10.0)
    print(f"Models endpoint status: {resp.status_code}")
    print(f"Response: {resp.text}\n")

# Test chat completions endpoint
chat_url = f"{base_url}/v1/chat/completions"
payload = {
    "model": "meta-llama/llama-3-3-70b-instruct",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ],
    "temperature": 0.7,
    "max_completion_tokens": 50
}

with httpx.Client() as client:
    resp = client.post(
        chat_url,
        headers=headers,
        json=payload,
        timeout=60.0
    )
    print(f"Chat completions status: {resp.status_code}")
    print(f"Response: {resp.text}")
