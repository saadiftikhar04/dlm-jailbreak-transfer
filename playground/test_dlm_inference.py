"""
Test script for DLM (Dialogue Language Model) inference
Tests: Llama3.2-3B-Instruct, Phi-3-mini, Qwen2.5-7B-Instruct
"""
import sys
sys.path.append('/scratch/si2356/projects/dlm-jailbreak-transfer')

from src.llm import llm_factory
import time

def test_dlm(model_name, prompt="Hello! Can you help me understand machine learning?"):
    """Test a single DLM"""
    print(f"\n{'='*80}")
    print(f"Testing DLM: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Initialize model
        print("Loading model...")
        start = time.time()
        model = llm_factory(model_name)
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Generate response with greedy decoding (safer)
        print(f"\nPrompt: {prompt}")
        print("Generating response (greedy decoding)...")
        start = time.time()
        response = model.generate(prompt, max_new_tokens=100, temperature=0.0)
        gen_time = time.time() - start
        
        print(f"\nResponse: {response[:300]}...")
        print(f"\n✓ Generated in {gen_time:.2f}s")
        print(f"✓ Response length: {len(response)} chars")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all DLMs"""
    test_prompt = "What is the capital of France? Answer briefly."
    
    dlms = [
        "llama3.2-3b-instruct",
        "phi-3.5-mini-instruct", 
        "qwen2.5-7b-instruct"
    ]
    
    results = {}
    
    for dlm_name in dlms:
        success = test_dlm(dlm_name, test_prompt)
        results[dlm_name] = "✓" if success else "✗"
    
    # Summary
    print(f"\n{'='*80}")
    print("DLM TEST SUMMARY")
    print(f"{'='*80}")
    for dlm, status in results.items():
        print(f"{status} {dlm}")
    
    print(f"\nTotal: {sum(1 for s in results.values() if s == '✓')}/{len(dlms)} passed")


if __name__ == "__main__":
    main()
