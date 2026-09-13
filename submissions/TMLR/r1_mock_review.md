# Review: Cross-Paradigm Jailbreak Transfer: A Unified Evaluation of Autoregressive and Diffusion Language Models

**Recommendation:** Reject in current form, major revision encouraged.
**Claims and evidence:** No (not yet adequately supported).
**Audience:** Yes.
**Certification:** No.

---

## 1. Summary of contributions

The paper evaluates three mechanism-distinct jailbreak attacks (PiF, token-salience flattening; ArrAttack, robust semantic rewriting; MetaCipher, cipher-based hidden-intent recovery) against six victim models: three causal or hybrid models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Falcon-H1R-7B) and three diffusion-family models (LLaDA-1.5-8B, Dream-v0-Instruct-7B, DiffuCoder-7B-Instruct), over a pooled 913-prompt benchmark drawn from HarmBench, MaliciousInstruct, JailbreakBench, and StrongREJECT. The evaluation uses a final-attempt convention in which each original prompt contributes exactly one attacked prompt and one final response, with ASR computed over each attack's own final evaluation stage.

The headline empirical claims are: (i) MetaCipher is the strongest attack overall (31.9% ASR) but sharply selective, succeeding at 71.5% / 71.1% / 47.3% on Qwen2.5, Llama, and LLaDA while staying below 1% on Falcon-H1R, Dream, and DiffuCoder; (ii) PiF (6.3%) and ArrAttack (7.1%) transfer poorly across both families; (iii) model-family labels are insufficient to predict jailbreak robustness, with a variance decomposition attributing 55.3% of ASR variance to models nested within family versus 6.1% to the family main effect. Analytically the paper introduces ACTR, mechanism exposure envelope (MEE), mechanism dependence gap (MDG), family sufficiency ratio (FSR), and vulnerability signatures, and proposes a post hoc four-gate account of jailbreak transfer.

---

## 2. Assessment

The question is well posed and timely, and the framing (transfer failure is not equivalent to intrinsic diffusion safety) is correct and worth establishing carefully. Several methodological choices are above the norm for this literature: the stage-explicit ASR denominator, the separation of official success verdicts from a descriptive failure categorizer, the explicit disclosure that the category taxonomy is attack-native, Wilson intervals on every cell, and the Appendix D reproduction audit with its unflattering numbers left in.

However, the evidence does not currently support the central claims at the level TMLR requires. My concerns cluster into four groups: (a) a missing capability control that makes "robustness" and "incapacity" indistinguishable for exactly the models on which the central negative result rests; (b) the absence of any experiment that separates the mechanisms the four-gate account posits; (c) non-comparability across attacks in judge, denominator, and decoding configuration, which is precisely the axis along which the paper's conclusions are drawn; and (d) an unvalidated reimplementation of two of the three attacks, both of which land near the floor. Each is addressable with modest additional compute, and I would be glad to see a revised version.

---

## 3. Strengths

**S1. The core question is the right one.** Distinguishing "this attack mechanism is incompatible with this decoding process" from "this model is safe" is the correct scientific decomposition, and the paper states it clearly and repeatedly.

**S2. Denominator discipline.** Equation 1 and Section 3.3 fix an ambiguity that is genuinely widespread in this literature. Refusing to substitute the best intermediate candidate for the final response is the right call and should be standard.

**S3. Judge separation.** Keeping the attack-specific official verdict as the ASR source while using a separate categorizer for failure analysis is a sound design, and the MetaCipher example (a binary harmfulness judge and a mechanism-specific judge disagreeing for principled reasons on partially decoded text) is a genuinely useful observation.

**S4. Honest self-audit.** Appendix D reports 41% row-level verdict agreement under regeneration and does not bury it. Section 5.3 discloses that the ArrAttack held-out stage was stratified by prompt source rather than content category, so cyber is 12.7% of that stage versus 7.0% of the pool. These disclosures are what make the paper's weaknesses diagnosable, and I want to credit them explicitly.

**S5. Internal arithmetic is consistent.** I checked the reported counts against the reported rates throughout Tables 1 and 2 and the Figure 3 support counts (including 646/913 = 70.8% and 141/165 = 85.5%, and 384 = 64 x 6 giving cyber as 7.0% of the pool). Everything reconciles. That is not universal in submissions of this type.

---

## 4. Weaknesses

### W1 (critical). No benign-capability control, so robustness and incapacity are confounded

Falcon-H1R records 0/913 under PiF, 0/165 under ArrAttack, and 2/913 under MetaCipher. Dream records 0/913 under PiF and 1/913 under MetaCipher. DiffuCoder, a code-specialized model, records 8/913 under MetaCipher. These six near-zero cells carry the paper's central negative claim, and the design cannot distinguish three explanations:

1. The model recognizes the request and refuses (safety).
2. The model cannot follow the attacked instruction at all, because the prompt is degenerate, the chat template is wrong, the generation is truncated, or the model's general instruction-following is weak (harness or capability artifact).
3. The judging pipeline discards the answer. Section A.3 notes Falcon-H1R emits a reasoning trace that is stripped before judging; if the stripping heuristic is imperfect, a compliant answer can be removed along with the trace.

An exact zero over 913 trials on two independent attack mechanisms is a strong enough null that it warrants harness validation rather than interpretation. The fix is standard: run a detoxified control set (the same prompts with the harmful object replaced by a benign one, or a general instruction-following set) through the identical harness, and report per-victim benign compliance rate, mean response length, empty or malformed output rate, and truncation rate. Until then, "Falcon-H1R is the most robust model in the portfolio" (Section 5.2) is not established, and neither is the low-susceptibility cluster.

### W2 (critical). The four-gate account is never tested, and the decisive experiment is cheap

Section 6.1 posits four gates and Section 6.2 concludes that Qwen2.5, Llama, and LLaDA "combine successful reconstruction with weak post-reconstruction arbitration" while Dream and DiffuCoder "rarely recover and follow the hidden harmful intent." That conjunction is exactly what the data cannot resolve: a 0.1% MetaCipher ASR on Dream is equally consistent with "cannot decode the cipher" and with "decodes fine and then refuses," and these have opposite defense implications.

The separating experiment is inexpensive and, in my view, would be the single largest scientific upgrade to the paper:

- **Gate 2 in isolation.** Apply the identical MetaCipher wrapper (same masking scheme, same substitution cipher, same silent-decode directive) to benign payloads and measure per-victim decode accuracy. This is a pure capability measurement with no safety confound.
- **Gate 3 in isolation.** Feed each victim the already-decoded plaintext harmful request directly and measure compliance. This is the model's arbitration behavior with reconstruction removed.

The 2x2 of (decode accuracy, plaintext compliance) would turn MetaCipher's selectivity from an observation into a mechanism, and would let the paper say whether LLaDA is a "cross-paradigm behavioral bridge" because of decoding ability or because of arbitration timing. Right now that phrase (Section 5.7) is an interpretation of a cosine similarity.

### W3 (critical). Cross-attack comparison is confounded by judge, denominator, and decoding configuration simultaneously

The paper's headline ordering (MetaCipher > ArrAttack > PiF) and its per-model MEE and MDG are computed across attacks that differ in three ways at once.

- **Judge.** PiF uses a DeepSeek binary harmfulness judge; MetaCipher uses a four-way categorical judge where only `compliance` counts; ArrAttack uses a separate DeepSeek binary judge. Appendix D supplies direct evidence that this matters: in 7 of 16 cases the recorded response itself is labeled harmful by the audit's binary judge while the official judge called it a failure. Judge boundary is therefore a first-order effect on the quantity being compared.
- **Denominator.** 913 for PiF and MetaCipher, 165 for ArrAttack.
- **Decoding.** Section A.3 states that LLaDA uses steps = gen length = block length = 128 under PiF but 128 steps over the full window under the other attacks. MDG is defined as max minus min ASR across attacks for a fixed model, so LLaDA's MDG of 40.3 points partly measures a decoding-configuration difference rather than attack dependence. Causal victims are greedy at temperature 0, Dream at 0.0, DiffuCoder at 0.3 with top-p 0.95, so family-level comparisons also carry a temperature confound.

Any one of these is manageable with a caveat. All three together, on the axis that generates the conclusions, is not.

### W4 (critical). The PiF and ArrAttack reimplementations are unvalidated, and both land near the floor

The PiF qualitative case in Figure 4 shows an attacked prompt that is lexically destroyed ("the step # by - step member group allows as with between for all c ..."), and the paper's own mechanism note concedes that "Gate 1 is never engaged." If the attacked prompts are word salad, then a 6.3% ASR measures reimplementation fidelity, not cross-paradigm transfer, and the RQ1 answer for PiF is uninterpretable. ArrAttack's 7.1% is similarly far below what its source paper claims, and Section 6.3 calls this "more surprising" without ruling out the implementation explanation.

The standard remedy is a positive control: reproduce each attack's reported ASR on at least one victim model from its own original paper, using this paper's harness and judge. If the reproduction matches, the low transfer numbers become a finding. If it does not, they are a bug report. This control is required before the negative results can be read as evidence about models rather than about code.

There is also an internal inconsistency to resolve. Sections 3.4 and 6.5 state that the 913 prompts are consumed "across training and validation" for ArrAttack, leaving 165 held out, while Section A.1 states that "no per-victim retraining of the rewriter or judge is performed" and that the pipeline uses off-the-shelf components (a T5 paraphraser, an MPNet encoder, a GPTFuzz classifier). If nothing is in fact trained on the 748 prompts, then ArrAttack can and should be evaluated on all 913, which dissolves W3's denominator problem entirely. If something is trained, please say what.

### W5 (major). Statistical support for RQ2 is under-specified and partly circular

RQ2 is the paper's most-quoted claim, and the variance decomposition in Section 4.3 is its strongest stated evidence. Several problems:

- **The reported components do not sum.** 27.9 + 6.1 + 55.3 = 89.3%. The remaining 10.7% is presumably the attack-by-model interaction, but that term is never named. This matters because the paper's qualitative thesis is precisely that mechanism-model compatibility dominates, and the interaction is the term that encodes it. Reporting the full ANOVA table and reconciling the narrative with the decomposition is necessary.
- **Model choice, not architecture, drives the result.** Each family contains one near-zero model (Falcon-H1R, Dream). With three models per family, dropping Falcon makes the causal family uniformly vulnerable and would likely reverse the conclusion. Please report leave-one-model-out sensitivity for the decomposition, the FSR, and the family gaps.
- **FSR below one is close to tautological here.** FSR(a) = Delta_a / R_a, where R_a averages within-family ranges. Given one near-zero model per family, the within-family range is large by construction, so FSR < 1 is nearly forced by the model selection rather than discovered. The paper already hedges FSR twice; I would cut it rather than hedge it a third time.
- **The model is inappropriate for the data.** Raw ASR percentages, many at or near the 0 boundary, with cell sizes of 913 and 165, analyzed by an unweighted ANOVA on proportions, violates homoscedasticity badly. A logistic mixed model on the case-level binary outcomes, for instance `success ~ attack * family + (1 | model) + (1 | prompt)`, would handle the boundary, the unequal denominators, and the repeated measurement of the same prompt across models in one step, and would give the family-effect claim an actual confidence statement.
- **Wilson intervals understate uncertainty.** They assume independent Bernoulli trials within a single run and therefore capture prompt sampling only, not the regeneration variance that Appendix D documents.

### W6 (major). The Appendix D audit is more damaging than its framing admits

Row-level verdict agreement of 21/51 (41%), with recorded successes reproducing as harmful in 14/21 and recorded failures holding in 14/30, is a serious instability. The decomposition into 7 judge-boundary cases and 9 regeneration cases is a good move, but two things follow that the paper does not draw out:

- Nine regeneration flips out of 51 sampled rows is roughly an 18% flip rate, which cannot be reconciled with a 0.1% or 0.9% MetaCipher ASR unless the stratified sampling is properly reweighted. Because the sample deliberately oversamples recorded successes (five per cell) relative to their base rate, the raw flip rates are not marginal escape rates. Please reweight by each cell's base rate and report an estimated marginal per-sample escape rate with an interval. That number, not the raw 47%, is what qualifies the low-ASR cells.
- Fifty-one rows drawn from six cells is too small a base for the qualification the paper asks it to carry. Multi-seed replication (three to five seeds on at least the six near-zero cells and the three high-ASR cells) would replace the audit's role and is the obvious next experiment.

### W7 (moderate). The novel metrics are largely arithmetic re-descriptions, and several are self-disclaimed

MEE is a max, MDG is a range, ACTR is a ratio, FSR is a ratio of a gap to a range, and the vulnerability signature is the raw row of the table. Of these, ACTR is declared non-interpretable at the floor (the ArrAttack value of 1.414 needs a footnote to prevent misreading), FSR is declared "a readable diagnostic rather than an inferential statistic," and cosine similarity is declared "magnitude-blind and compressed in the positive orthant" and demoted to a "descriptive neighbor check." The LLaDA-Qwen2.5 cosine of 0.999 illustrates the problem: any two MetaCipher-dominated three-vectors in the positive orthant will be near-collinear, so the number is close to uninformative.

Contributions 2, 3, and 5 in the introduction are all versions of "we defined some ratios over a 3 x 6 table." I would consolidate to two contributions (the cross-paradigm measurement, and the stage-explicit protocol), keep MEE and MDG as two extra columns in Table 1 without acronyms, keep the variance decomposition as the actual evidence for RQ2, and drop ACTR, FSR, and cosine similarity entirely. This would also free roughly a page for the experiments requested above.

### W8 (moderate). Content-conditioned analysis compares populations, not mechanisms

The paper discloses that the taxonomy is attack-native and that ArrAttack's held-out stage drifts in composition, then plots the three attacks as adjacent bars per category in Figure 3 anyway, which invites exactly the cross-attack reading the caveat forbids. Since all 913 prompts come from a common pool, labeling them once with a single classifier (or with StrongREJECT's own categories) and re-plotting would make the figure mean what it appears to mean. Additionally, the pooled per-category rates average over three models with near-zero ASR, so MetaCipher's 44.5% on cyber understates the rate on susceptible models by roughly a factor of two. Per-family or per-model category breakdowns for the three susceptible models would be more informative.

### W9 (moderate). Prompt pool overlap is not addressed

400 + 100 + 100 + 313 = 913 exactly, which implies no deduplication was performed. JailbreakBench behaviors overlap substantially with HarmBench and AdvBench-derived sets. Duplicates inflate the effective sample size, correlate the benchmark-level comparison in Section 5.3, and violate the independence assumption behind the Wilson intervals. Please report near-duplicate statistics (for instance, embedding similarity above a stated threshold) and either deduplicate or report ASR with clustered intervals.

### W10 (moderate). No diffusion-native attack is run, in a diffusion-focused paper

The paper argues correctly and at length (Sections 2.4, 6.4, Limitations) that transfer failure is not intrinsic safety, and cites DIJA and parallel-decoding attacks as the reason. But the claim that Dream and DiffuCoder are vulnerable to diffusion-native attacks is imported from prior work rather than measured here, and Gate 4 is explicitly not probed. One diffusion-native baseline on the three diffusion victims would convert the paper's most important interpretive claim into a result, and would give the four-gate account its missing fourth column. Given that DIJA is public, I consider this a reasonable ask rather than a scope expansion.

### W11 (minor). Human validation as described cannot support the word "validated"

A single non-blind author annotator, no preserved confusion matrix, no per-stratum agreement, and no kappa. The paper is candid about this, but contribution 4 in the introduction still advertises a "human-validated protocol." Either re-run the 597-case audit with preserved labels and a second annotator on a subset (a few hundred double-annotated cases would suffice for kappa), or remove the word from the contribution list and describe it as a spot check.

### W12 (minor). Reproducibility details

- Exact model identifiers and Hugging Face revisions for all six victims are not given. "Falcon-H1R-7B" in particular is described only as "the public Falcon-H1 reasoning-tuned 7B checkpoint used in our runs."
- `max_new_tokens` is stated only for diffusion victims (512-token window). Generation length interacts directly with compliance judgments, since a truncated procedural answer may read as incomplete to a judge. Please report the setting and the truncation rate for every victim.
- No code link for the three reimplementations. The stated release artifact is a metadata-and-verdict package; the reimplementation code is what W4 makes essential.
- Judge calls use "the default API sampling settings of the corresponding attack implementation," which means the judges are themselves stochastic. Please set temperature 0 for judging or report judge self-agreement on a repeated subset.

### W13 (minor). Organization

Section 3 ("Task Definition") and Section 4 ("Methodology") overlap substantially; the seven-stage pipeline in Section 4.1 restates Figure 1, which itself diagrams a fairly standard evaluation loop. Compressing these by a page or more would cost nothing and make room for the experiments above.

---

## 5. Requested changes

**Critical (required for me to move toward acceptance):**

1. Add a benign-capability control per victim and report benign compliance, empty or malformed rate, mean length, and truncation rate. Explicitly validate the Falcon-H1R reasoning-trace stripping. (W1)
2. Add the Gate 2 / Gate 3 separation experiment: cipher decode accuracy on benign payloads, and plaintext compliance on decoded requests, per victim. (W2)
3. Re-judge all 11,946 cases with a single common judge (StrongREJECT rubric or the HarmBench classifier) and report a second ASR table alongside the official-judge table. State whether the attack ordering survives. (W3)
4. Add positive controls reproducing PiF's and ArrAttack's originally reported ASR on at least one victim from each attack's own paper. (W4)
5. Report PiF and MetaCipher restricted to ArrAttack's 165-prompt held-out set as a matched-denominator comparison, or, if nothing is actually trained for ArrAttack, evaluate ArrAttack on the full 913. Resolve the Section 3.4 versus Section A.1 inconsistency. (W3, W4)
6. Fix the LLaDA decoding inconsistency across attacks, or exclude LLaDA from MEE and MDG until it is fixed. (W3)
7. Replace the ANOVA on percentages with a case-level logistic mixed model, report the full decomposition including the interaction term, and add leave-one-model-out sensitivity for every family-level conclusion. (W5)
8. Reweight the Appendix D audit by cell base rates to give a marginal escape-rate estimate, and add multi-seed replication for at least the six near-zero cells and the three high-ASR cells. (W6)

**Recommended:**

9. Run one diffusion-native attack on the three diffusion victims. (W10)
10. Unify the content taxonomy across attacks and re-plot Figure 3; add per-family category rates. (W8)
11. Cut ACTR, FSR, and cosine similarity; consolidate the five contributions into two or three. (W7)
12. Report near-duplicate statistics across the four benchmark suites. (W9)
13. Re-run or re-scope the human validation claim. (W11)

**Minor:**

14. Exact model IDs and revisions, generation length for all victims, judge temperature, code release. (W12)
15. Merge Sections 3 and 4. (W13)

---

## 6. Questions for the authors

1. For the six near-zero cells, what fraction of final responses are explicit refusals, versus generic or off-target replies, versus empty or malformed output? The categorizer already produces these labels, so this table should be immediately available and would go a long way toward addressing W1.
2. What is the mean and distribution of semantic similarity between the original prompt x and the attacked prompt x' under PiF? The MPNet encoder is already in the ArrAttack pipeline. If PiF's similarity distribution is far below what the original paper reports, that resolves W4 for PiF directly.
3. Was any component of ArrAttack fit on the 748 non-held-out prompts, and if so which one?
4. Does the attack ordering (MetaCipher > ArrAttack > PiF) survive under a single common judge, on a subsample if the full re-judge is expensive?
5. What is the remaining 10.7% of variance in the Section 4.3 decomposition, and was ASR transformed before the decomposition?
6. For MetaCipher, what fraction of Dream and DiffuCoder responses are classified `wrong decryption` versus `refusal`? This is the fastest partial proxy for the Gate 2 / Gate 3 question in W2 and may already exist in the judged CSVs.
7. Were the four benchmark suites deduplicated?

---

## 7. Broader impact concerns

None that block publication. The ethics section is appropriately scoped: no new attack method, no new harmful corpus, no victim weights, truncated and sanitized qualitative examples, and a stated intention to release verdicts and metadata rather than operational completions. The dual-use argument (vulnerability signatures are diagnostic for defenders and add no capability beyond the original attack papers) is reasonable for this class of work.

Two suggestions. First, please state the access-control mechanism for the response fields concretely, since "sanitized or access-controlled" currently covers two quite different policies. Second, Figure 4's MetaCipher panel is more operationally specific than the other two; the step structure of the SQL injection answer is visible even truncated. Consider trimming it further or replacing the payload with a clearly non-functional placeholder.

---

## 8. Overall

The paper asks a question worth answering and gets several methodological details right that this literature routinely gets wrong. What is missing is a set of controls, each individually modest, that would separate the paper's conclusions from the alternatives: capability from robustness, decoding ability from arbitration, mechanism from judge, and finding from implementation. With W1 through W4 addressed I would expect this to be a solid contribution, and the four-gate framing in particular could become a useful organizing device for the subfield rather than a post hoc narrative.

---

*Note on conflicts: MetaCipher (Chen et al., 2025) is one of the three evaluated attacks and is central to the paper's headline result. Anyone assigned to review this submission who is an author of that work should declare the conflict to the AE.*
