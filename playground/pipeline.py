"""
Complete Experimental Pipeline for DLM Jailbreak Evaluation

This script implements the full pipeline:
1. Load datasets (HarmBench, JailbreakBench, StrongREJECT, MaliciousInstruct)
2. Run jailbreak methods (PiF, ArrAttack, MetaCipher)
3. Evaluate results with appropriate judges
4. Generate summary statistics

Usage:
    python pipeline.py --model <model_path> --dataset <dataset_name> --method <method_name>
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm_jailbreaks import (
    JailbreakConfig, PiFAttack, ArrAttack, MetaCipherAttack,
    run_experiment, load_dataset
)
from judges import run_judge


class ExperimentPipeline:
    """Main experimental pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config.get('results_dir', './results'))
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def load_config(self, config_path: str = None) -> dict:
        """Load experimental configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'results_dir': './results',
                'data_dir': './data',
                'models': {
                    'dlm_victim': 'path/to/dlm/model',
                },
                'datasets': ['harmbench', 'jailbreakbench', 'strongreject', 'maliciousinstruct'],
                'methods': ['pif', 'arrattack', 'metacipher'],
                'judges': {
                    'harmbench': 'harmbench',
                    'jailbreakbench': 'jailbreakbench',
                    'strongreject': 'strongreject',
                    'maliciousinstruct': 'maliciousinstruct',
                },
            }
            
    def load_model(self, model_path: str):
        """Load DLM model and tokenizer"""
        print(f"Loading model from {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto'
        )
        
        return model, tokenizer
        
    def run_attack(self, model, tokenizer, dataset_name: str, method: str, 
                   num_samples: int = None) -> str:
        """
        Run a single attack experiment
        
        Returns path to output CSV
        """
        # Load dataset
        print(f"\nLoading {dataset_name} dataset...")
        harmful_prompts = load_dataset(dataset_name, self.config.get('data_dir', './data'))
        
        if num_samples:
            harmful_prompts = harmful_prompts[:num_samples]
            
        print(f"Loaded {len(harmful_prompts)} prompts")
        
        # Configure attack
        if method == 'pif':
            config = JailbreakConfig(
                method='pif',
                num_steps=100,
                pif_k=10,
                pif_alpha=0.1
            )
        elif method == 'arrattack':
            config = JailbreakConfig(
                method='arrattack',
                arr_num_samples=20,
                arr_temperature=1.0
            )
        elif method == 'metacipher':
            # Try multiple cipher types
            cipher_type = 'caesar'  # Can iterate through: caesar, atbash, vigenere, etc.
            config = JailbreakConfig(
                method='metacipher',
                cipher_type=cipher_type,
                include_instructions=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Run experiment
        print(f"Running {method} attack on {dataset_name}...")
        results_df = run_experiment(model, tokenizer, harmful_prompts, method, config)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"{dataset_name}_{method}_{timestamp}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return str(output_path)
        
    def run_judge_evaluation(self, dataset_name: str, results_csv: str) -> str:
        """
        Run judge evaluation on attack results
        
        Returns path to judged CSV
        """
        judge_type = self.config['judges'].get(dataset_name, dataset_name)
        
        output_path = results_csv.replace('.csv', '_judged.csv')
        
        print(f"\nRunning {judge_type} judge on {results_csv}...")
        run_judge(judge_type, results_csv, output_path)
        
        return output_path
        
    def generate_summary(self, judged_results: list):
        """Generate summary statistics across all experiments"""
        summary = {
            'experiments': [],
            'overall_stats': {}
        }
        
        for result_path in judged_results:
            df = pd.read_csv(result_path)
            
            # Extract metadata from filename
            filename = Path(result_path).stem
            parts = filename.split('_')
            dataset = parts[0]
            method = parts[1]
            
            # Calculate stats
            if 'label' in df.columns:
                success_rate = (df['label'] == 'Successful').sum() / len(df) * 100
                stats = {
                    'dataset': dataset,
                    'method': method,
                    'num_samples': len(df),
                    'success_rate': success_rate,
                    'successful': (df['label'] == 'Successful').sum(),
                    'failed': (df['label'] == 'Failed').sum(),
                }
            elif 'score' in df.columns:
                stats = {
                    'dataset': dataset,
                    'method': method,
                    'num_samples': len(df),
                    'mean_score': df['score'].mean(),
                    'median_score': df['score'].median(),
                    'std_score': df['score'].std(),
                }
            else:
                stats = {
                    'dataset': dataset,
                    'method': method,
                    'num_samples': len(df),
                }
                
            summary['experiments'].append(stats)
            
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSummary saved to {summary_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        df_summary = pd.DataFrame(summary['experiments'])
        print(df_summary.to_string(index=False))
        
        return summary_path
        
    def run_full_pipeline(self, model_path: str, datasets: list = None, 
                         methods: list = None, num_samples: int = None):
        """
        Run complete experimental pipeline
        
        Args:
            model_path: Path to DLM model
            datasets: List of datasets to evaluate (default: all configured)
            methods: List of methods to test (default: all configured)
            num_samples: Number of samples per dataset (default: all)
        """
        # Load model
        model, tokenizer = self.load_model(model_path)
        
        # Use configured datasets/methods if not specified
        datasets = datasets or self.config['datasets']
        methods = methods or self.config['methods']
        
        judged_results = []
        
        # Run experiments
        for dataset in datasets:
            for method in methods:
                print(f"\n{'='*80}")
                print(f"Experiment: {dataset} + {method}")
                print(f"{'='*80}")
                
                try:
                    # Step 1: Run attack
                    results_csv = self.run_attack(
                        model, tokenizer, dataset, method, num_samples
                    )
                    
                    # Step 2: Judge results
                    judged_csv = self.run_judge_evaluation(dataset, results_csv)
                    judged_results.append(judged_csv)
                    
                except Exception as e:
                    print(f"Error in experiment {dataset}+{method}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        # Step 3: Generate summary
        if judged_results:
            self.generate_summary(judged_results)
            
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)


def run_single_experiment(model_path: str, dataset: str, method: str, 
                         num_samples: int = None, output_dir: str = './results'):
    """
    Run a single experiment (convenience function)
    
    Example:
        run_single_experiment(
            model_path='path/to/dlm',
            dataset='harmbench',
            method='metacipher',
            num_samples=100
        )
    """
    config = {
        'results_dir': output_dir,
        'data_dir': './data',
        'judges': {
            'harmbench': 'harmbench',
            'jailbreakbench': 'jailbreakbench',
            'strongreject': 'strongreject',
            'maliciousinstruct': 'maliciousinstruct',
        }
    }
    
    # Save temp config
    config_path = '/tmp/exp_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
        
    pipeline = ExperimentPipeline(config_path)
    pipeline.run_full_pipeline(model_path, [dataset], [method], num_samples)


def main():
    parser = argparse.ArgumentParser(description='DLM Jailbreak Experimental Pipeline')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to DLM model')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--dataset', type=str, nargs='+',
                       choices=['harmbench', 'jailbreakbench', 'strongreject', 
                               'maliciousinstruct', 'all'],
                       default=['all'],
                       help='Dataset(s) to evaluate')
    parser.add_argument('--method', type=str, nargs='+',
                       choices=['pif', 'arrattack', 'metacipher', 'all'],
                       default=['all'],
                       help='Jailbreak method(s) to test')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples per dataset (default: all)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Handle 'all' selections
    all_datasets = ['harmbench', 'jailbreakbench', 'strongreject', 'maliciousinstruct']
    all_methods = ['pif', 'arrattack', 'metacipher']
    
    datasets = all_datasets if 'all' in args.dataset else args.dataset
    methods = all_methods if 'all' in args.method else args.method
    
    # Create pipeline
    pipeline = ExperimentPipeline(args.config)
    
    # Run experiments
    pipeline.run_full_pipeline(
        model_path=args.model,
        datasets=datasets,
        methods=methods,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
