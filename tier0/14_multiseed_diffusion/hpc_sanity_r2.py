"""HPC sanity for R2 Step 14: verify diffucoder loads + generates ONE row.
Independent, no exec gymnastics. Run with raven python on a GPU node.
"""
import os, sys, json, time, signal
os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")
sys.path.insert(0, "/scratch/bc3194/dlm-jailbreak-transfer/scripts")
import torch
from pif_target_models import load_target, target_generate

SRC = "/scratch/bc3194/dlm-jailbreak-transfer/tier0/14_multiseed_diffusion"
with open(os.path.join(SRC, "multiseed_sample_r2.json")) as f:
    rows = json.load(f)
row = [r for r in rows if r["model"] == "diffucoder"][0]
print("diffucoder attack:", row["attack"], "prompt len:", len(row["attacked_prompt"]), flush=True)

t0 = time.time()
model, tokenizer = load_target("diffucoder", offline=False)
model.eval()
print("loaded diffucoder in %.1fs" % (time.time() - t0), flush=True)

torch.manual_seed(1); torch.cuda.manual_seed_all(1)
class _Timeout(Exception): pass
def _alarm(sig, frm): raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
signal.alarm(180)
try:
    resp = target_generate(model, tokenizer, "diffucoder", row["attacked_prompt"], max_new_tokens=512)
    print("GEN OK %.1fs resp_len=%d" % (time.time() - t0, len(resp)), flush=True)
    print("resp head:", resp[:120], flush=True)
except _Timeout:
    print("GEN TIMEOUT after 180s", flush=True)
except Exception as e:
    print("GEN FAIL %.1fs: %s: %s" % (time.time() - t0, type(e).__name__, str(e)[:200]), flush=True)
finally:
    signal.alarm(0)
print("SANITY COMPLETE", flush=True)