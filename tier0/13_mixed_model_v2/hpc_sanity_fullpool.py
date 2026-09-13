# R2 Step 13 — HPC sanity: syntax + utils import (no model load)
import ast, importlib.util, sys, os

# 1. AST-parse stage5_fullpool.py (syntax check only, don't exec)
p = "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool/stage5_fullpool.py"
src = open(p, encoding="utf-8").read()
try:
    ast.parse(src)
    print("AST_PARSE_OK", p)
except SyntaxError as e:
    print("AST_PARSE_FAIL", e)
    sys.exit(1)

# 2. Try importing qwen_utils + model_utils from the fullpool tree (no exec of stage5)
sys.path.insert(0, "/scratch/bc3194/dlm-jailbreak-transfer/arrattack_fullpool")
try:
    import utils.qwen_utils as qu
    import model_utils as mu
    print("IMPORT_OK")
    print("  qwen_utils PROJECT_DIR =", qu.PROJECT_DIR)
    print("  qwen_utils SNAPSHOT_BASE =", qu.SNAPSHOT_BASE)
    print("  model_utils HF_CACHE =", mu.HF_CACHE)
    print("  model_utils REG llama =", mu.MODEL_REGISTRY["llama"])
    print("  model_utils REG dream =", mu.MODEL_REGISTRY["dream"][:80])
except Exception as e:
    import traceback; traceback.print_exc()
    print("IMPORT_FAIL", e)
    sys.exit(1)