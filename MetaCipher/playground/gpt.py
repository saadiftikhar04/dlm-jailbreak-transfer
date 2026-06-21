from openai import OpenAI
import os


def main():
    # Get the OpenAI API Key from environment. It is recommended to add an export line in ~/.bashrc, so that it is automatically loaded at every login.
    client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

    # Initialize prompts
    # Prompt 1: A prompt with overtly malicious content (bomb). You should expect to be rejected. 
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    # Prompt 2: An adversarial prompt with ciphered malicious content (bomb)
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()
    # Prompt 3: An improved prompt for a hard questions
    with open("../prompts/examples/text/self_harm_acrostic.txt", "r") as file:
        prompt_3 = file.read()

    # List all available models
    models = client.models.list()
    for model in models:
        print(model.id)

    # gpt-4o-2024-11-20, o1-mini-2024-09-12
    response_with_limit = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "user",
                "content": prompt_3
            }
        ],
        # temperature=0.0
        # max_tokens=200
    )
    print(response_with_limit.choices[0].message.content)


if __name__=="__main__":
    main()