#!/usr/bin/env python3
"""
Comprehensive Sweep Script for Jailbreak Experiments
Runs experiments across multiple models, datasets, and attack methods
"""
import sys
sys.path.append('/scratch/si2356/projects/dlm-jailbreak-transfer')

import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import itertools


class ExperimentSweep:
    """Manages sweeps across multiple experimental configurations."""
    
    # Available configurations
    MODELS = {
        'llm': [
            'llama3.2-3b-instruct',
            'llama3.1-8b-instruct',
            'mistral-7b-instruct-v0.3',
            'qwen2.5-7b-instruct',
        ],
        'dlm': [
            'meta-llama/Llama-3.2-3B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct', 
            'mistralai/Mistral-7B-Instruct-v0.3',
            'Qwen/Qwen2.5-7B-Instruct',
        ]
    }
    
    DATASETS = [
        'jailbreakbench',
        'advbench',
        'harmbench',
    ]
    
    ATTACKS = {
        'baseline': 'scripts/experiments/run_baseline.py',
        'pif': 'scripts/experiments/run_pif.py',
        'gcg': 'scripts/experiments/run_gcg.py',
        'arrattack': 'scripts/experiments/run_arrattack.py',
        'metacipher': 'scripts/experiments/test/metacipher.py',
    }
    
    def __init__(self, output_dir: Path = None, judge_type: str = 'keyword'):
        """
        Initialize sweep manager.
        
        Args:
            output_dir: Base directory for results
            judge_type: Type of judge to use for evaluation
        """
        self.output_dir = output_dir or Path('results/sweeps')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judge_type = judge_type
        
        # Create sweep log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sweep_id = f"sweep_{timestamp}"
        self.log_file = self.output_dir / f"{self.sweep_id}.log"
        self.results_file = self.output_dir / f"{self.sweep_id}_results.csv"
        
        self.results = []
    
    def log(self, message: str):
        """Log message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
    
    def run_experiment(
        self,
        attack: str,
        model: str,
        dataset: str,
        num_prompts: int = 100,
        extra_args: Dict = None
    ) -> Dict:
        """
        Run a single experiment configuration.
        
        Args:
            attack: Attack method name
            model: Model name
            dataset: Dataset name
            num_prompts: Number of prompts to test
            extra_args: Additional arguments for the attack script
            
        Returns:
            Dict with experiment results
        """
        extra_args = extra_args or {}
        
        # Build command
        script = self.ATTACKS[attack]
        cmd = ['python', script]
        
        # Add common arguments
        if attack == 'baseline':
            cmd.extend(['--model', model])
        else:
            cmd.extend(['--victim', model])
        
        cmd.extend([
            '--dataset', dataset,
            '--num-prompts', str(num_prompts),
        ])
        
        # Add attack-specific arguments
        for key, value in extra_args.items():
            cmd.extend([f'--{key}', str(value)])
        
        self.log(f"\n{'='*80}")
        self.log(f"Running: {attack} on {model} with {dataset}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"{'='*80}")
        
        # Run experiment
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            success = result.returncode == 0
            output = result.stdout
            error = result.stderr if not success else ""
            
        except subprocess.TimeoutExpired:
            success = False
            output = ""
            error = "Experiment timed out after 2 hours"
        except Exception as e:
            success = False
            output = ""
            error = str(e)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Find results file
        results_file = self._find_results_file(attack, model, dataset)
        
        # Evaluate results if successful
        asr = None
        num_successful = None
        if success and results_file and results_file.exists():
            asr, num_successful = self._evaluate_results(results_file)
        
        # Record results
        result_dict = {
            'sweep_id': self.sweep_id,
            'timestamp': datetime.now().isoformat(),
            'attack': attack,
            'model': model,
            'dataset': dataset,
            'num_prompts': num_prompts,
            'success': success,
            'duration_seconds': duration,
            'asr': asr,
            'num_successful': num_successful,
            'results_file': str(results_file) if results_file else None,
            'error': error if not success else None,
        }
        
        self.results.append(result_dict)
        
        # Save intermediate results
        self._save_results()
        
        if success:
            self.log(f"✓ Success! ASR: {asr:.2%}, Duration: {duration:.1f}s")
        else:
            self.log(f"✗ Failed: {error}")
        
        return result_dict
    
    def _find_results_file(self, attack: str, model: str, dataset: str) -> Optional[Path]:
        """Find the results file for an experiment."""
        if attack == 'baseline':
            base_dir = Path('results/baseline')
        else:
            base_dir = Path(f'results/{attack}')
        
        # Try different possible paths
        candidates = [
            base_dir / dataset / f"{model}.csv",
            base_dir / f"{dataset}_{model}.csv",
            base_dir / dataset / f"{model}_results.csv",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _evaluate_results(self, results_file: Path) -> tuple:
        """
        Evaluate results using the judge.
        
        Returns:
            Tuple of (asr, num_successful)
        """
        try:
            # Run judge
            cmd = [
                'python', 'scripts/evaluation/judge.py',
                '--results', str(results_file),
                '--judge-type', self.judge_type,
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse ASR from output
            for line in result.stdout.split('\n'):
                if 'Attack Success Rate' in line or 'ASR:' in line:
                    # Extract percentage
                    import re
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        asr = float(match.group(1)) / 100
                        
                        # Also try to get num successful
                        match2 = re.search(r'(\d+)/(\d+)', line)
                        if match2:
                            num_successful = int(match2.group(1))
                            return asr, num_successful
                        
                        # Fallback: calculate from ASR
                        df = pd.read_csv(results_file)
                        num_successful = int(asr * len(df))
                        return asr, num_successful
            
            # Fallback: read the judged file
            judged_file = results_file.parent / f"{results_file.stem}_judged.csv"
            if judged_file.exists():
                df = pd.read_csv(judged_file)
                if 'jb_success' in df.columns:
                    asr = df['jb_success'].mean()
                    num_successful = df['jb_success'].sum()
                    return asr, num_successful
        
        except Exception as e:
            self.log(f"Warning: Could not evaluate results: {e}")
        
        return None, None
    
    def _save_results(self):
        """Save current results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
    
    def run_sweep(
        self,
        attacks: List[str],
        models: List[str],
        datasets: List[str],
        num_prompts: int = 100,
        model_type: str = 'llm',
        extra_args: Dict = None,
    ):
        """
        Run sweep across multiple configurations.
        
        Args:
            attacks: List of attack methods
            models: List of models (or 'all' for all models of type)
            datasets: List of datasets (or 'all' for all datasets)
            num_prompts: Number of prompts per experiment
            model_type: 'llm' or 'dlm'
            extra_args: Additional arguments for attack scripts
        """
        # Resolve 'all' shorthand
        if models == ['all']:
            models = self.MODELS[model_type]
        if datasets == ['all']:
            datasets = self.DATASETS
        if attacks == ['all']:
            attacks = list(self.ATTACKS.keys())
        
        # Generate all combinations
        configs = list(itertools.product(attacks, models, datasets))
        total = len(configs)
        
        self.log(f"\n{'='*80}")
        self.log(f"Starting sweep: {self.sweep_id}")
        self.log(f"Total experiments: {total}")
        self.log(f"Attacks: {attacks}")
        self.log(f"Models: {models}")
        self.log(f"Datasets: {datasets}")
        self.log(f"{'='*80}\n")
        
        # Run all experiments
        for i, (attack, model, dataset) in enumerate(configs, 1):
            self.log(f"\n[{i}/{total}] Starting: {attack} + {model} + {dataset}")
            
            try:
                self.run_experiment(
                    attack=attack,
                    model=model,
                    dataset=dataset,
                    num_prompts=num_prompts,
                    extra_args=extra_args
                )
            except Exception as e:
                self.log(f"Error running experiment: {e}")
                # Record failed experiment
                self.results.append({
                    'sweep_id': self.sweep_id,
                    'timestamp': datetime.now().isoformat(),
                    'attack': attack,
                    'model': model,
                    'dataset': dataset,
                    'num_prompts': num_prompts,
                    'success': False,
                    'error': str(e),
                })
                self._save_results()
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of sweep results."""
        df = pd.DataFrame(self.results)
        
        self.log(f"\n{'='*80}")
        self.log(f"SWEEP COMPLETE: {self.sweep_id}")
        self.log(f"{'='*80}")
        
        # Overall stats
        total = len(df)
        successful = df['success'].sum()
        failed = total - successful
        
        self.log(f"\nOverall Results:")
        self.log(f"  Total experiments: {total}")
        self.log(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        self.log(f"  Failed: {failed} ({failed/total*100:.1f}%)")
        
        # ASR by attack method
        if 'asr' in df.columns:
            self.log(f"\nASR by Attack Method:")
            asr_by_attack = df[df['asr'].notna()].groupby('attack')['asr'].agg(['mean', 'std', 'count'])
            for attack, row in asr_by_attack.iterrows():
                self.log(f"  {attack:15s}: {row['mean']*100:5.1f}% ± {row['std']*100:4.1f}% (n={int(row['count'])})")
        
        # ASR by model
        if 'asr' in df.columns:
            self.log(f"\nASR by Model:")
            asr_by_model = df[df['asr'].notna()].groupby('model')['asr'].agg(['mean', 'std', 'count'])
            for model, row in asr_by_model.iterrows():
                self.log(f"  {model:30s}: {row['mean']*100:5.1f}% ± {row['std']*100:4.1f}% (n={int(row['count'])})")
        
        # ASR by dataset
        if 'asr' in df.columns:
            self.log(f"\nASR by Dataset:")
            asr_by_dataset = df[df['asr'].notna()].groupby('dataset')['asr'].agg(['mean', 'std', 'count'])
            for dataset, row in asr_by_dataset.iterrows():
                self.log(f"  {dataset:15s}: {row['mean']*100:5.1f}% ± {row['std']*100:4.1f}% (n={int(row['count'])})")
        
        self.log(f"\nResults saved to: {self.results_file}")
        self.log(f"Log saved to: {self.log_file}")
        self.log(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run sweep across multiple jailbreak experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all attacks on all LLM models with JailbreakBench
  python scripts/experiments/sweep.py --attacks all --models all --datasets jailbreakbench
  
  # Run baseline and PiF on specific models
  python scripts/experiments/sweep.py --attacks baseline pif --models llama3.2-3b-instruct mistral-7b-instruct-v0.3
  
  # Quick test with small number of prompts
  python scripts/experiments/sweep.py --attacks baseline --models llama3.2-3b-instruct --num-prompts 10
  
  # Full sweep (WARNING: Takes many hours!)
  python scripts/experiments/sweep.py --attacks all --models all --datasets all --num-prompts 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--attacks',
        nargs='+',
        default=['baseline'],
        choices=list(ExperimentSweep.ATTACKS.keys()) + ['all'],
        help='Attack methods to test'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['llama3.2-3b-instruct'],
        help='Models to test (or "all" for all models)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['jailbreakbench'],
        choices=ExperimentSweep.DATASETS + ['all'],
        help='Datasets to test'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model-type',
        type=str,
        default='llm',
        choices=['llm', 'dlm'],
        help='Type of models (llm or dlm)'
    )
    
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=100,
        help='Number of prompts per experiment'
    )
    
    parser.add_argument(
        '--judge-type',
        type=str,
        default='keyword',
        choices=['keyword', 'llm'],
        help='Type of judge for evaluation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for sweep results'
    )
    
    # Parse known args to allow pass-through
    args, unknown = parser.parse_known_args()
    
    # Parse extra args for attack scripts
    extra_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                extra_args[key] = unknown[i + 1]
                i += 2
            else:
                extra_args[key] = True
                i += 1
        else:
            i += 1
    
    # Create sweep manager
    sweep = ExperimentSweep(
        output_dir=args.output_dir,
        judge_type=args.judge_type
    )
    
    # Run sweep
    sweep.run_sweep(
        attacks=args.attacks,
        models=args.models,
        datasets=args.datasets,
        num_prompts=args.num_prompts,
        model_type=args.model_type,
        extra_args=extra_args if extra_args else None
    )


if __name__ == '__main__':
    main()
