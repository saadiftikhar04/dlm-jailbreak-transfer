"""
Test script for LLM inference
Tests: Qwen3-8B, Falcon-H1R-7B, Claude-Sonnet-4.5
"""
import sys
sys.path.append('/scratch/si2356/projects/dlm-jailbreak-transfer')

from src.llm import llm_factory
import time

def test_model(model_name, prompt="Hello, how are you?"):
    """Test a single model"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Initialize model
        print("Loading model...")
        start = time.time()
        model = llm_factory(model_name)
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Generate response
        print(f"\nPrompt: {prompt}")
        print("Generating response...")
        start = time.time()
        response = model.generate(prompt, max_new_tokens=256, temperature=0.7)
        gen_time = time.time() - start
        
        print(f"\nResponse: {response}")
        print(f"\n✓ Generated in {gen_time:.2f}s")
        print(f"✓ Response length: {len(response)} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all LLMs"""
    test_prompt = "What is machine learning? Explain briefly."
    
    models = [
        "qwen3-8b",
        "falcon-h1r-7b",
        "claude-sonnet-4.5"
    ]
    
    results = {}
    
    for model_name in models:
        success = test_model(model_name, test_prompt)
        results[model_name] = "✓" if success else "✗"
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model, status in results.items():
        print(f"{status} {model}")
    
    print(f"\nTotal: {sum(1 for s in results.values() if s == '✓')}/{len(models)} passed")


if __name__ == "__main__":
    main()
