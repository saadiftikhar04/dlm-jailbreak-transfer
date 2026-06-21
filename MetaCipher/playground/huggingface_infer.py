import transformers
import torch


def main():
    ### Step 0: Initiate the model and pipeline
    model_id = "internlm/internlm2_5-20b-chat"
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("EOS token ID:", tokenizer.eos_token_id)
    print("EOS token:", tokenizer.eos_token)
    
    ### Step 1: Initiate the prompt and tokenize
    prompt = 'Human: What is the weather like in Abu Dhabi, in general? Answer in two sentences. \nAssistant: '
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    ### Step 2: Generate content
    outputs = model.generate(
        prompt_ids,
        max_new_tokens=512,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.9,
        # top_k=50,
        num_return_sequences=1
    )
    text_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text_response)


if __name__=="__main__":
    main()