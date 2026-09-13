"""
Self-contained LLaDA generation — no dllm package required.
Inlines: BaseAlphaScheduler, LinearAlphaScheduler, get_num_transfer_tokens, generate.
"""
import dataclasses, math
from typing import ClassVar, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers

Number = Union[float, torch.Tensor]

# ── Scheduler ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BaseAlphaScheduler:
    __registry__: ClassVar[dict] = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlphaScheduler.__registry__[cls.__name__] = cls
    def __call__(self, t):
        return self.alpha(t)
    def alpha(self, i):
        i_t = torch.as_tensor(i, dtype=torch.float32)
        return self._alpha(i_t)
    def _alpha(self, i): raise NotImplementedError
    def reverse_mask_prob(self, s, t):
        return (1 - self(torch.as_tensor(s, dtype=torch.float32))) / \
               (1 - self(torch.as_tensor(t, dtype=torch.float32)))

@dataclasses.dataclass
class LinearAlphaScheduler(BaseAlphaScheduler):
    def _alpha(self, i): return 1 - i

# ── Token schedule ────────────────────────────────────────────────────────────

def get_num_transfer_tokens(mask_index, steps, scheduler, stochastic=False):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps-1, -1, -1), range(steps)):
            s /= steps; t /= steps
            rtp = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * rtp
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, rtp).sample().to(torch.int64))
    return num_transfer_tokens

# ── Gumbel noise ──────────────────────────────────────────────────────────────

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)

# ── Generate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model, tokenizer, prompts,
    scheduler=None,
    steps=128, max_new_tokens=256, max_length=1024,
    block_length=128, temperature=0.0,
    cfg_scale=0.0, cfg_keep_tokens=None,
    remasking="random", return_dict_in_generate=False,
    stochastic_transfer=False,
):
    if scheduler is None:
        scheduler = LinearAlphaScheduler()

    assert 1 <= block_length <= max_new_tokens
    assert 1 <= steps

    mask_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids('<|mdm_mask|>')
    eos_id  = tokenizer.eos_token_id
    prompt_lens = [p.shape[0] for p in prompts]

    if max_new_tokens:
        max_length = max_new_tokens + max(prompt_lens)
    else:
        max_new_tokens = max_length - max(prompt_lens)

    B, T = len(prompts), max_length
    x = torch.full((B, T), eos_id, dtype=torch.long, device=model.device)
    for i, p in enumerate(prompts):
        x[i, :prompt_lens[i]] = p
        x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = mask_id

    unmasked_index = (x != mask_id) & (x != eos_id)
    if cfg_keep_tokens:
        keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=model.device))
        unmasked_index = unmasked_index & ~keep_mask

    num_blocks = math.ceil(max_new_tokens / block_length)
    steps = math.ceil(steps / num_blocks)

    for b in range(num_blocks):
        block_mask_index = torch.zeros((B, block_length), dtype=torch.bool, device=x.device)
        for j in range(B):
            start = prompt_lens[j] + b * block_length
            end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                w = end - start
                block_mask_index[j, :w] = (x[j, start:end] == mask_id)

        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps, scheduler, stochastic_transfer)

        for i in range(num_transfer_tokens.size(1)):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone(); un_x[unmasked_index] = mask_id
                logits, un_logits = torch.chunk(model(torch.cat([x, un_x], 0)).logits, 2, 0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            if remasking == "low_confidence":
                x0_p = torch.gather(F.softmax(logits, -1), -1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
            else:
                raise NotImplementedError(remasking)

            for j in range(B):
                x0_p[j, prompt_lens[j] + (b+1)*block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    if not return_dict_in_generate:
        return x
    return {"sequences": x}
