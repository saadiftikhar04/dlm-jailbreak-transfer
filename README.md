# Jailbreak Attack Runners - Delivery Package

**Date**: January 2025  
**Purpose**: Complete setup and runner scripts for PiF, ArrAttack, and GCG attacks  
**For**: MetaCipher Jailbreak Transferability Analysis

## What's Included

This package contains everything needed to run comprehensive jailbreak attack experiments on the MetaCipher platform.

### 📋 Documentation (5 files)

<<<<<<< HEAD
1. **QUICK_START.md**
=======
1. **QUICK_START.md** ⭐ START HERE
>>>>>>> 6c7c78e (Update repository with latest changes from Jubail)
   - 5-minute setup guide
   - Essential commands only
   - Troubleshooting basics

2. **SETUP_README.md** 
   - Complete setup instructions
   - Detailed usage for each runner
   - Troubleshooting section
   - Expected results

3. **DEPLOYMENT_CHECKLIST.md**
   - Step-by-step deployment guide
   - Pre-deployment checklist
   - Monitoring and management
   - Common issues and solutions

4. **PACKAGE_SUMMARY.md**
   - Technical details
   - Design principles
   - Integration guide
   - Performance characteristics

5. **experiments.yaml**
   - Sample configuration
   - All attacks/datasets/victims
   - Customizable parameters

### 🔧 Setup Scripts (2 files)

1. **setup_hpc.sh**
   - Creates conda environment
   - Installs all dependencies
   - Downloads NLTK data
   - Generates .env template
   - **Run this first!**

2. **organize_repo.sh**
   - Organizes file structure
   - Creates directories
   - Makes scripts executable
   - **Run this second!**

### 🎯 Attack Runners (3 files)

1. **run_pif.py** - PiF Attack
   - Flattens attention distribution
   - Injects neutral words
   - Uses synonyms and prefixes
   - Fast, black-box

2. **run_arrattack.py** - ArrAttack
   - Adaptive robust rewriting
   - 8 rewriting strategies
   - Robustness judging
   - LLM-based

3. **run_gcg.py** - GCG Attack
   - Gradient-guided optimization
   - Adversarial suffix generation
   - Transfer attacks
   - GPU-accelerated

### 🎮 Control Scripts (3 files)

1. **run_all_experiments.py**
   - Master experiment runner
   - Runs all combinations
   - Configuration support
   - Progress tracking

2. **quick_test.py**
   - Quick validation
   - Tests all runners
   - Single-prompt tests
   - Debug mode

3. **validate_environment.py**
   - Environment checker
   - Validates setup
   - Checks dependencies
   - Colorized output

## File Sizes

```
Total: ~130KB (scripts + documentation)
+ 50GB (models cache after setup)
+ 10GB (results after experiments)
```

## Quick Deploy

```bash
# 1. Upload to HPC
scp -r delivery_package/ user@hpc:/workspace/

# 2. SSH and setup
ssh user@hpc
cd /workspace/delivery_package/
./setup_hpc.sh

# 3. Organize
./organize_repo.sh

# 4. Configure
nano .env  # Add API keys

# 5. Validate
conda activate metacipher
python validate_environment.py

# 6. Test
python quick_test.py

# 7. Run experiments
python run_all_experiments.py --config experiments.yaml
```

## What You'll Get

After running experiments:

### Results Structure
```
results/
├── pif/
│   ├── jailbreakbench/
│   │   ├── claude-sonnet-4.5.csv
│   │   ├── llama2-70b-chat.csv
│   │   └── ...
│   └── ...
├── arrattack/
│   └── ...
└── gcg/
    └── ...
```

### CSV Schema
Each CSV contains:
- `prompt`: Original malicious prompt
- `category`: Prompt category
- `victim_response`: Model's response
- `success`: Attack success (boolean)
- `time`: Time taken
- Attack-specific columns

## Requirements

### Minimal
- Python 3.10+
- 50GB storage
- 16GB RAM
- Internet access

### Recommended
- GPU (for GCG)
- 64GB RAM
- 8+ CPU cores
- SLURM cluster

### API Keys (at least one)
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude)
- Google API key (for Gemini)

## Documentation Guide

| Document | When to Read |
|----------|--------------|
| QUICK_START.md | First time setup |
| SETUP_README.md | Need detailed info |
| DEPLOYMENT_CHECKLIST.md | HPC deployment |
| PACKAGE_SUMMARY.md | Technical details |

## Expected Timeline

- **Setup**: 30 minutes
- **Testing**: 15 minutes
- **Single experiment**: 1-4 hours
- **Full suite**: 6-12 days (parallel) or 150-300 hours (sequential)

## Support

1. **Check documentation**: All info is in the MD files
2. **Run validation**: `python validate_environment.py`
3. **Check logs**: `logs/` directory
4. **Verbose mode**: Add `--log-level DEBUG` to any command

## Integration

These scripts integrate with:
- ✓ Existing `src/llm.py` model factory
- ✓ Existing `src/utils.py` utilities
- ✓ Existing dataset format (`processed.csv`)
- ✓ Existing MetaCipher runner
- ✓ Existing post-processing judges

No modifications to existing code needed!

## Verification Checklist

After deployment, verify:
- [ ] `setup_hpc.sh` completed successfully
- [ ] `validate_environment.py` shows all ✓
- [ ] `quick_test.py` passes for at least one attack
- [ ] Results CSV created in `results/` directory
- [ ] Logs created in `logs/` directory

## Next Steps

1. **Read**: QUICK_START.md
2. **Setup**: Run setup_hpc.sh
3. **Test**: Run quick_test.py
<<<<<<< HEAD
=======
4. **Deploy**: Run experiments
5. **Analyze**: Use post-processing scripts

## Contact

For issues:
1. Check troubleshooting in SETUP_README.md
2. Review logs in `logs/` directory
3. Run with `--verbose` or `--log-level DEBUG`

---

**Ready to deploy?** Start with `QUICK_START.md`!

**Package Version**: 1.0  
**Delivered**: January 2025  
**Status**: Production-ready ✓
>>>>>>> 6c7c78e (Update repository with latest changes from Jubail)
