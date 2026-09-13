"""R2 Step 16: LLaDA decoding-configuration sensitivity sweep.

LLaDA in the main experiment is generated via a CUSTOM masked iterative
denoising loop (pif_target_models.llada branch): block_length=128, and
steps/block varies by attack (PiF => 128 steps/block, ArrAttack/MetaCipher
=> 32 steps/block since steps=gen_length but gen is capped). Step 16 quantifies
how much remasking strategy / step budget / temperature moves LLaDA's output
(and thus its ASR) by re-denosing a sample of prompts under several configs.

CLI: python sweep_llada_r2.py --dataset data.json --out out.jsonl --config c
Configs (c): base(128s/block,t=0) | steps32(t0) | steps256(t0) | temp05(128s,t=.5)
         | temp10(128s,t=1.0) | conflict remedy: vary the block-level remask
         schedule. Each config writes a JSONL of {id, config, response, status}.

Implements the exact llada loop but with (block_length, steps_per_block,
temperature) exposed as parameters. One process, resident model. Resumable.
"""
import os, sys, json, time, argparse
import torch, torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/scratch/bc3194/huggingface_cache")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
from pif_target_models import load_target

_ADD_GUMBEL = None  # re-impl below


def add_gumbel(logits, temperature):
    if temperature == 0:
        return logits
    noise = torch.zeros_like(logits).uniform_().clamp(1e-9, 1)
    return logits + (-(-noise.log()).log()) * temperature


def llada_sweep_generate(model, tokenizer, prompt, gen_length=512,
                         block_length=128, steps_per_block=32, temperature=0.0,
                         seed=1):
    """Replicate the pif_target_models llada loop with overridable config."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    MASK_ID = tokenizer.mask_token_id or 126336
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.cuda()
    attn_mask = enc.attention_mask.cuda()
    prompt_len = input_ids.shape[1]
    gen_length = max(128, (min(gen_length, 512, max(2048 - prompt_len, 128)) // 128) * 128)
    num_blocks = gen_length // block_length
    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long).cuda()
    x[:, :prompt_len] = input_ids.clone()
    with torch.no_grad():
        for nb in range(num_blocks):
            bs = prompt_len + nb * block_length
            be = prompt_len + (nb + 1) * block_length
            nmask = (x[:, bs:be] == MASK_ID).sum(dim=1)
            spb = steps_per_block
            ntt = torch.zeros(1, spb, device=x.device, dtype=torch.long) + (nmask // spb)
            ntt[0, :(nmask % spb)] += 1
            for i in range(spb):
                mask_index = (x == MASK_ID)
                logits = model(x).logits
                x0 = torch.argmax(add_gumbel(logits, temperature), dim=-1)
                x0_p = F.softmax(logits.float(), dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, be:] = float("-inf")
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))
                _, sel = torch.topk(conf[0], k=int(ntt[0, i]))
                trans = torch.zeros_like(x0, dtype=torch.bool)
                trans[0, sel] = True
                x[trans] = x0[trans]
    generated = x[0, prompt_len:]
    ids = generated.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text.split("<|dlm_pad|>")[0].replace("<|im_end|>", "").strip()


CONFIGS = {
    # name: (block_length, steps_per_block, temperature)
    "base128s": (128, 128, 0.0),     # PiF-equivalent (128 steps/block)
    "base32s":  (128, 32, 0.0),      # ArrAttack/MC-equivalent (32 steps/block)
    "steps256": (128, 256, 0.0),     # more denoising
    "steps64":  (128, 64, 0.0),
    "temp05":   (128, 128, 0.5),
    "temp10":   (128, 128, 1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(args.dataset) as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    model, tokenizer = load_target("llada", offline=False)
    model.eval()
    blk, spb, temp = CONFIGS[args.config]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["id"])
    todo = [it for it in items if str(it.get("id", it.get("prompt_idx", ""))) not in done]
    print(f"[{args.config}] {len(todo)} to run (cfg: bl={blk} spb={spb} t={temp})", flush=True)
    if not todo:
        print("already done"); return
    open_f = open(args.out, "a")
    ok = 0
    try:
        for it in todo:
            pid = str(it.get("id", it.get("prompt_idx", "")))
            prompt = it.get("attacked_prompt") or it.get("prompt") or it.get("wrapped_prompt")
            try:
                resp = llada_sweep_generate(model, tokenizer, prompt,
                                            block_length=blk, steps_per_block=spb,
                                            temperature=temp, seed=args.seed)
                status = "ok"; ok += 1
            except Exception as e:
                resp = ""; status = f"error:{type(e).__name__}:{str(e)[:150]}"
            rec = {"id": pid, "config": args.config, "response": resp, "status": status,
                   "seed": args.seed}
            open_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); open_f.flush()
            if ok % 10 == 0:
                print(f"  [{args.config}] ok={ok}", flush=True)
    finally:
        open_f.close()
    print(f"wrote {args.out}: new={len(todo)} ok={ok}", flush=True)


if __name__ == "__main__":
    main()