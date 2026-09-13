"""R2 Step 13 — local Falcon 1-prompt smoke (verify env is sound before full run)."""
import os, sys
os.environ.setdefault("HF_HOME", "/home/bc3194/Desktop/huggingface_cache")
os.environ["TARGET"] = "falcon"
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, "falcon_env")
sys.path.insert(0, ENV)

import ast, time
# parse stage5_fullpool → verify DATASET_CONFIG
src = open(os.path.join(ENV, "stage5_fullpool.py"), encoding="utf-8").read()
tree = ast.parse(src)
print("AST_OK stage5_fullpool")

# Import qwen_utils + model_utils resolve
import model_utils, utils.qwen_utils as qu
print("PROJECT_DIR:", qu.PROJECT_DIR)
print("model_utils HF_CACHE:", model_utils.HF_CACHE)
print("REG falcon:", model_utils.MODEL_REGISTRY["falcon"])

# load falcon (respect local cache; may take ~30-60s)
t0 = time.time()
m, tok = model_utils.load_model("falcon", offline=True)
m.eval()
print(f"falcon loaded in {time.time()-t0:.1f}s", flush=True)
print("SMOKE_IMPORT_OK")