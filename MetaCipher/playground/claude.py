import anthropic
import os


def main():
    # Get the Anthropic Token. This is a required environment variable.
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  

    # Initialize prompts
    # Prompt 1: A prompt with overtly malicious content (bomb). You should expect to be rejected. 
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    # Prompt 2: An adversarial prompt with ciphered malicious content (bomb)
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()
    # Prompt 3: Ciphered prompt with distraction
    with open("../prompts/examples/text/bomb_long_acrostic.txt", "r") as file:
        prompt_3 = file.read()
        
    # claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt_3}],
        temperature=0.0,
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()