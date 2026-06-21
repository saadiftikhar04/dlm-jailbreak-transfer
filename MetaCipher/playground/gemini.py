import os
import google.generativeai as genai
from google.generativeai import types



def list_all_models():
    print("Available Gemini Models:")
    models = genai.list_models()
    for model in models:
        # Only show models that support text generation
        if "generateContent" in model.supported_generation_methods:
            print(f" - {model.name}")


def main():
    google_token = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=google_token)

    # List available models
    list_all_models()

    # Load prompts
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()

    # Choose model
    model = genai.GenerativeModel("gemini-2.0-flash")
    generation_config = {
        "max_output_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["\n\n\n"]
    }
    response = model.generate_content(prompt_2, generation_config=generation_config)
    print(response.text)


if __name__ == "__main__":
    main()
