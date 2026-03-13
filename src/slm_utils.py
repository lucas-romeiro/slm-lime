import requests

def query_slm(prompt, model = "ai/qwen2.5"):
    url = "http://localhost:12434/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = response.json()

    return data
