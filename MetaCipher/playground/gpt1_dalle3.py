from openai import OpenAI
import os
import base64


def main():
    # Get the OpenAI Token. I hard-coded my path to the file that saves it. You can change it to your path, or simply put your key here. 
    openai_token = os.getenv('OPENAI_API_KEY')

    # Initialize prompts
    # Prompt 0: A fully normal prompt
    prompt_0 = "Draw me a photo-realistic image of a lovely cat."
    # Prompt 1: A prompt with a deduction test
    prompt_1 = "What is the letter after g? Draw that in capital please. " 
    # # Prompt 2: A prompt with overtly malicious content (gun). You should expect to be rejected. 
    # with open("../prompts/examples/gun.txt", "r") as file:
    #     prompt_2 = file.read()
    # # Prompt 3: An adversarial prompt with ciphered malicious content (gun)
    # with open("../prompts/examples/gun_acrostic.txt", "r") as file:
    #     prompt_3 = file.read()

    # Get response
    client = OpenAI(api_key=openai_token)
    # List all available models
    models = client.models.list()
    print("Available image generation models:")
    for model in models.data:
        print(model.id)
    
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt_0
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open("gpt1_native_image.png", "wb") as f:
        f.write(image_bytes)


if __name__=="__main__":
    main()