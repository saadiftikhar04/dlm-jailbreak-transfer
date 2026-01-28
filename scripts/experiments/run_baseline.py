"""
Baseline Experiment Runner
Run models on malicious prompt datasets WITHOUT any jailbreak methods
"""
import sys
sys.path.append('/scratch/si2356/projects/dlm-jailbreak-transfer')

import argparse
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from src.llm import llm_factory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, data_dir: Path, num_prompts: int = None) -> pd.DataFrame:
    """Load dataset and standardize columns."""
    
    # Map dataset names to file paths
    dataset_paths = {
        'jailbreakbench': data_dir / 'jailbreakbench' / 'processed.csv',
        'harmbench': data_dir / 'harmbench' / 'processed.csv',
        'advbench': data_dir / 'advbench' / 'harmful_behaviors.csv',
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(dataset_paths.keys())}")
    
    dataset_path = dataset_paths[dataset_name]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Standardize column names
    if dataset_name == 'advbench':
        # AdvBench uses 'goal' instead of 'prompt'
        if 'goal' in df.columns:
            df['prompt'] = df['goal']
        # Add default category if missing
        if 'category' not in df.columns:
            df['category'] = 'harmful'
    
    # Ensure required columns
    if 'prompt' not in df.columns:
        raise ValueError(f"Dataset missing 'prompt' column")
    
    if 'category' not in df.columns:
        logger.warning("Dataset missing 'category' column, adding default")
        df['category'] = 'unknown'
    
    # Limit number of prompts if specified
    if num_prompts:
        df = df.head(num_prompts)
        logger.info(f"Limited to {num_prompts} prompts")
    
    logger.info(f"Loaded {len(df)} prompts")
    return df


def run_baseline(
    model_name: str,
    dataset_name: str,
    num_prompts: int = None,
    data_dir: Path = Path('dataset'),
    output_dir: Path = Path('results/baseline'),
    max_new_tokens: int = 256,
    temperature: float = 0.0
):
    """Run baseline experiment (no jailbreak) on a model and dataset."""
    
    logger.info("="*80)
    logger.info("Baseline Experiment Runner")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Max tokens: {max_new_tokens}")
    logger.info(f"Temperature: {temperature}")
    
    # Load dataset
    df = load_dataset(dataset_name, data_dir, num_prompts)
    
    # Initialize model
    logger.info(f"Initializing model: {model_name}...")
    model = llm_factory(model_name)
    logger.info("Model ready")
    
    # Prepare output
    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name.replace('/', '_')}.csv"
    
    # Run inference
    results = []
    logger.info("Starting baseline inference...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Baseline"):
        prompt = row['prompt']
        category = row.get('category', 'unknown')
        
        try:
            start_time = time.time()
            response = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            elapsed_time = time.time() - start_time
            
            results.append({
                'index': idx,
                'prompt': prompt,
                'response': response,
                'category': category,
                'time': elapsed_time,
                'success': True
            })
            
            logger.info(f"Prompt {idx+1}/{len(df)}: Generated in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error on prompt {idx}: {str(e)}")
            results.append({
                'index': idx,
                'prompt': prompt,
                'response': f"ERROR: {str(e)}",
                'category': category,
                'time': 0,
                'success': False
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Summary
    logger.info("="*80)
    logger.info("Baseline Experiment Summary")
    logger.info("="*80)
    logger.info(f"Total prompts: {len(results)}")
    logger.info(f"Successful: {sum(r['success'] for r in results)}")
    logger.info(f"Failed: {sum(not r['success'] for r in results)}")
    logger.info(f"Average time: {sum(r['time'] for r in results)/len(results):.2f}s")
    logger.info(f"Completed at {datetime.now()}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., qwen3-8b, llama3.2-3b-instruct)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['jailbreakbench', 'harmbench', 'advbench'],
                        help='Dataset to use')
    parser.add_argument('--num-prompts', type=int, default=None,
                        help='Number of prompts to test (default: all)')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Generation temperature')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='results/baseline',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_baseline(
        model_name=args.model,
        dataset_name=args.dataset,
        num_prompts=args.num_prompts,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
