from llm_sdk import call_ollama_api


data = {
    "model": "qwen3:0.6b",
    "prompt": "Hello!",
    "stream": False,
    "think": False
}


def main():
    result = call_ollama_api(data)
    print(result)


if __name__ == "__main__":
    main()
