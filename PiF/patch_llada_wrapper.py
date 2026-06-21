"""
Patches run_pif.py DLMWrapper to use official LLaDA masked diffusion generate loop.
Based on: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
mask_id = 126336 (official)
"""

with open("/scratch/si2356/dlm-jailbreak-transfer/PiF/run_pif.py") as f:
    src = f.read()

old = '''# ── DLM wrapper: makes diffusion models compatible with repo's .generate() ────
class DLMWrapper(torch.nn.Module):
    """
    Repo calls: tgt_model.generate(input_ids=..., max_length=512)
    DLMs expose: model.diffusion_generate(...)
    This bridges them transparently.
    """
    def __init__(self, model, tokenizer, model_key="llada"):
        super().__init__()
        self._model     = model
        self._tok       = tokenizer
        self._model_key = model_key

    def generate(self, input_ids, max_length=512, **kwargs):
        from pif_target_models import target_generate
        prompt_len = input_ids.shape[1]
        max_new    = max(64, max_length - prompt_len)
        # Use target_generate which correctly handles LLaDA-1.5 (no diffusion_generate)
        prompt_text = self._tok.decode(input_ids[0], skip_special_tokens=False)
        resp = target_generate(self._model, self._tok, self._model_key,
                               prompt_text, max_new_tokens=max_new)
        # Re-encode full response and return as tensor matching repo expectations
        full = self._tok(prompt_text + resp, return_tensors="pt").input_ids.to(input_ids.device)
        return full

    def to(self, *a, **kw): return self
    def eval(self):         return self
    def parameters(self):   return self._model.parameters()
    # allow attribute access for hasattr checks in repo
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)'''

new = '''# ── Official LLaDA masked diffusion sampling (from ML-GSAI/LLaDA/generate.py) ─
def _llada_add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def _llada_get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def _llada_generate(model, prompt, attention_mask=None,
                    steps=128, gen_length=128, block_length=128,
                    temperature=0., remasking='low_confidence', mask_id=126336):
    """Official LLaDA iterative masked diffusion sampling loop."""
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length),
                       dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end   = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = _llada_get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = _llada_add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                x0_p = torch.rand((x.shape[0], x.shape[1]), device=x.device)

            # only unmask positions that are currently masked
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-torch.inf, device=x.device))

            # pick top-k confident tokens per block to unmask
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for j in range(x.shape[0]):
                blk_s = prompt.shape[1] + num_block * block_length
                blk_e = prompt.shape[1] + (num_block + 1) * block_length
                n = num_transfer_tokens[j, i].item()
                if n > 0:
                    block_conf = confidence[j, blk_s:blk_e]
                    _, top_idx = torch.topk(block_conf, k=int(n))
                    transfer_index[j, blk_s + top_idx] = True

            x = torch.where(transfer_index, x0, x)

    return x


# ── DLM wrapper: makes diffusion models compatible with repo's .generate() ────
class DLMWrapper(torch.nn.Module):
    """
    Repo calls: tgt_model.generate(input_ids=..., max_length=512)
    DLMs: use official masked diffusion sampling loop (LLaDA/generate.py).
    """
    MASK_ID = 126336  # official LLaDA mask token id

    def __init__(self, model, tokenizer, model_key="llada"):
        super().__init__()
        self._model     = model
        self._tok       = tokenizer
        self._model_key = model_key

    def generate(self, input_ids, max_length=512, **kwargs):
        prompt_len = input_ids.shape[1]
        gen_length = max(64, max_length - prompt_len)
        # Round gen_length to multiple of block_length (32)
        block_length = 32
        gen_length = ((gen_length + block_length - 1) // block_length) * block_length
        steps = min(gen_length, 128)
        # steps must be divisible by num_blocks
        num_blocks = gen_length // block_length
        steps = max(num_blocks, (steps // num_blocks) * num_blocks)

        attn = torch.ones_like(input_ids)
        out = _llada_generate(
            self._model, input_ids,
            attention_mask=attn,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            remasking='low_confidence',
            mask_id=self.MASK_ID,
        )
        # return only generated part shaped as [1, prompt_len + gen_length]
        return out

    def to(self, *a, **kw): return self
    def eval(self):         return self
    def parameters(self):   return self._model.parameters()
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)'''

assert old in src, f"Pattern not found! Check if previous patch changed the class."
src = src.replace(old, new)

with open("/scratch/si2356/dlm-jailbreak-transfer/PiF/run_pif.py", "w") as f:
    f.write(src)
print("Patched successfully")
