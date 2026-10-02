# TMLR R3 — RESUME CHECKPOINT (2026-10-02)

READ THIS FIRST after the session gap. Everything below is verified on-disk as of 2026-10-02.

属性/惯例：回复用户用中文；代码仓库(本 repo)由用户手动 commit，我绝不代 commit；只有 Overleaf_TMLR(嵌套 LaTeX 论文 repo)按惯例无条件 commit+push。论文当前编译 29 页，0 error / 0 undefined / 0 overfull。所有修改要遵守用户偏好：不用 em/en-dash(散文中用逗号/冒号)、美式拼写、.json 一律 indent=4、"revision 文档更新同步在 plan 里打 ✅"。

## HARD REQUIREMENTS (before touching anything)
1. NESTED REPO: Overleaf_TMLR 是嵌套 git repo(remote=git.overleaf.com)。任何 Overleaf 编辑前先 `git -C Overleaf_TMLR pull origin main`。编辑后：latexmk 验证 → `git add <具体文件>` → `git -c user.email="bc3194@nyu.edu" -c user.name="Boyuan Chen" commit` → `git push origin main` → 报告 SHA 范围。不要 `git add -A`(会带入构建产物，除非确认 .gitignore 已覆盖)。
2. latexmk 验证用：`cd Overleaf_TMLR && latexmk -pdf -interaction=nonstopmode main.tex`，检查 0 undefined / 0 overfull / 0 error。若跑 `latexmk -c` 会清掉 .bbl 造成假 undefined + 损坏 pdf——修复 = `rm -f main.aux main.bbl main.blg main.log main.out main.pdf && pdflatex && bibtex && pdflatex && pdflatex`(完整三遍)。
3. 主 repo(本目录)从不 commit。工作树里未跟踪/已修改的内容提交与否交给用户。

## 项目状态（会话目标：TMLR R3 整稿修订，基于 r3_mock_review）

### Overleaf 已推送提交(最新在前，到 HEAD 7a7d2a7)
- `7a7d2a7` Specified the corresponding author as the review-period access administrator and the post-review deletion policy for full field-level responses.
- `e893276` Reported the earlier LLaDA pilot sweep as 6/25 versus 1/25 with Wilson intervals and an explicit small-sample qualification.
- `4c9da43` Followed the user's T4 ruling: removed unreconciled full-pool ArrAttack ASRs and kept the verified 165-prompt estimate as the only reportable ASR; added the row-key and missing-verdict explanation.
- `4c9da43` Followed the user's T4 ruling: removed unreconciled full-pool ArrAttack ASRs and kept the verified 165-prompt estimate as the only reportable ASR; added the row-key and missing-verdict explanation.
- `ddfa69a` Replaced the Figure 3 grouped bars with a within-attack content table, added source-verified counts and Wilson intervals to the category and benchmark claims, and reconciled table references.
- `d3eef34` Added the T2 matched-step LLaDA bracket to the appendix, corrected the discussion cross-reference, and regenerated Table 1 from the validated control/count sources.
- `ff026cf` Reconciled reproduction-audit concordance definitions and values.
- `d3841ef` Recomputed Table 1 Wilson CIs/lifts from raw counts and precise controls; added control/common-judge Wilson CIs; corrected T6 precision and T14 seed summary; added T3 Qwen/Llama seed rows
- `a1aa2f4` Followed user scope rulings: no vendor-contact discussion or COI assertion in the manuscript; judge-circularity caveat remains
- `a1aa2f4` Followed user scope rulings: no vendor-contact discussion or COI assertion in the manuscript; judge-circularity caveat remains
- `9c869db` Avoids reporting a proportion without Wilson intervals in the low-positive PiF caveat; reports raw counts instead
- `f3f5245` T16 interim judge-as-reference confusion table + T23 allocated GPU-hours; later scope ruling supersedes its COI sentence and vendor wording
- `6444ff0` T19: appendix 加 pinned_checkpoints 小节，pin 全部 7 个 victim/attack HF git-sha(取自 HPC runtime cache)+ judge 访问细节
- `1a87ab7` T20: Figure3 重绘为 within-attack panels(审稿 M3)，caption 更新；新 PDF 已部署到 images/prompt_group_bars_final.pdf
- `c61c060`/`1ebf5f6` T18: conclusion + limitations 去 judge-dependence 冗余
- `30034a3` T21: §3.4 ArrAttack staging 措辞与附录一致(transfer-only)
- `c67cbd4` T7-5: 配对 McNemar 替换 Wilson 重叠判定
- `edb345c` T10: 51-row audit 重写为 spot-check
- `2536034` T7/T8/T9/T11: family claim 撤回(模型级不确定性)、judge-invariant 主证据、Table1 Falcon no-answer 分母、四门 benign per-gate

### 哈希(已写进 appendix，运行 cache 实测；如需复核：HPC HF cache 每个 models--*/snapshots/ 目录)
Llama-3.1-8B `0e9e39f249a16976918f6564b8830bc894c89659` · Llama-3.2-3B `0cb88a4f764b7a12671c53f0838cd831a0843b95` · Qwen2.5-7B `a09a35458c702b33eeacc393d103063234e8bc28` · Falcon-H1R-7B `a6f74bf181389908efd6970878d7ee2b42f5d417` · LLaDA-1.5-8B `84346fd91ba60252d260022201ad6fc5a3468fb2` · Dream-v0-7B `05334cb9faaf763692dcf9d8737c642be2b2a6ae` · DiffuCoder-7B `4fdd4580064ca5d11808069ce78f88d068753c96`

### r3_plan.txt 状态(逐项 ✅ 权威来源)
已完成打勾: BLOCK0(D1-D4) / BLOCK1(T1✅,T2✅,T3✅,T4✅) / BLOCK2(T6-T11✅，T5由用户决定豁免) / BLOCK3(T12-T17✅; T16使用judge-as-reference并明确非ground truth) / BLOCK4(T18-T23✅)。
DiffuCoder 碎片输出 = 稳定真实模型行为(plaintext 300 条也碎片化;3 seed 各 600 条统计完全一致 empty=85)。数据无需重跑。

## 当前阻塞与下一步
1. [WAIVED BY USER] T5 human annotation is omitted; 46 existing labels are retained as an incomplete audit and are not treated as a human reference. Do not assign further annotation to the user.
2. [✅] T23 GPU-hours: Slurm accounting for 91 GPU allocations in the project working directory from 2026-08-11 through 2026-09-17 gives 1,411.77 allocated GPU-hours (rounded to 1,412), on A100 40/80 GB and H100 GPUs; this is allocated wall time, not device utilization.
3. [USER RULING] T24 resolved without vendor outreach: no vendor was contacted, so no dates exist; the prior prospective-disclosure statement was removed, and any reviewer response should cite the unfilled-template finding and state that Claude-positive results are not treated as confirmed susceptibility. T25: the corresponding author listed in OpenReview administers request-based sentence-verification access during review, then revokes access and deletes full field-level responses; only non-operational metadata, verdicts, aggregates, and sanitized examples remain. T26: no formal COI assertion solely due to MetaCipher coauthorship and no identifying statement in the blind PDF; factually disclose the non-financial author overlap through OpenReview competing-interest metadata at submission for editorial assessment.
4. [COMPLETE] User authorized T2/T3 runs on 2026-10-02. T3 Qwen/Llama generation and judging are complete; both T2 counterfactual generations and paired common-judge analyses are complete. The HPC checkout was an older `main` snapshot (`db563c0`) without a configured remote and contained 22 untracked artifacts; only required files were synced, preserving those artifacts.
5. T3 Llama: 75 matched prompts, 3 seeds, all generations valid. DeepSeek binary judge at temperature 0, base-rate reweighted ASR = 8.980%, 12.181%, 13.089%; mean 11.417%, range 4.110 pp, variance 3.10676 pp². Wilson intervals are recorded in `tier0/18_t3_multiseed_llama/t3_seed_summary.csv`.
6. T3 Qwen: 70 matched prompts, 3 seeds, all generations and judgments valid. Weighted ASR = 72.951%, 74.349%, 68.650%; mean 71.983%, range 5.699 pp, variance 5.88033 pp². Wilson intervals for all T3 seed estimates are in `tier0/18_t3_multiseed_llama/t3_seed_summary.csv`.
7. T2 paired LLaDA step bracket (n=50/attack, common DeepSeek binary judge at temp 0): PiF 128→32 steps 6.008%→2.017%, lift −3.991 pp, McNemar p=.625; MetaCipher 32→128 steps 10.240%→46.769%, lift +36.529 pp, p=8e−6. Per-row scores and weighted Wilson summaries: `tier0/16_llada_sensitivity/t2_bracket_scores.csv`, `t2_bracket_summary.csv`.
7. Numeric audit found and corrected two analysis bugs: T1 CSV columns had rate/interval values interleaved under grouped headers, and T14 seed aggregation omitted `attack` from the join key, causing PiF to reuse ArrAttack judgments. Table 1/common-judge/T3/T14 values now reconcile to their on-disk sources. DOD 14 remains partial while the remaining manuscript numeric inventory is completed.
8. Overleaf is pushed and clean at `7a7d2a7` (`origin/main` matches; ahead count 0). Final `latexmk -pdf -interaction=nonstopmode main.tex` completed at 29 pages with no errors, undefined references, or overfull boxes; existing underfull and PDF-version warnings remain. T2 is in Appendix `sec:appendix_t2_bracket` and the discussion points there. Table 1 generator syntax and output were verified; repeated generation is byte-identical, and its separate attack-transfer summary table is preserved byte-for-byte. Its source script remains an uncommitted main-repo change for the user to review.
9. DOD 14 numeric coverage added on 2026-10-02: §5.1 across-model standard deviations recalculated from the 18-cell matrix; ANOVA shares and cited mixed-model terms matched to source reports; §5.3 benchmark counts/rates/CIs recomputed from all 11,946 official rows; MetaCipher native-category counts/rates/CIs recomputed from 5,478 raw judged rows; ArrAttack's too-general prompt type verified as 14/78 (17.9%, Wilson CI [11.0,27.9]); all 18 Table 4 group cells recomputed from raw HPC judged CSVs and now report counts with Wilson intervals. The table caption notes repeated-prompt dependence and attack-specific taxonomies. DOD 14 remains partial while the remaining manuscript numeric inventory is checked.
10. T4 ruling (user chose withdrawal 2026-10-02): the full-pool strict-judge percentages are not reportable because result/cache keys mix numeric global indices and normalized prompts, deduplicating the raw records yields inconsistent counts, and GPT-sol is missing 40/5,380 verdicts. The 165-prompt 7.1% protocol-faithful estimate remains the only reportable ArrAttack ASR. The manuscript now removes the stale full-pool point estimates and explains the audit/selection limitation; no additional judge calls.
11. T25 resolved per user confirmation 2026-10-02: the corresponding author listed in OpenReview administers request-based, time-limited access during review for sentence-level verification, revokes access and deletes full field-level responses at review close, while metadata/verdicts/aggregates/sanitized examples remain. No raw-response access administrator is needed after deletion.

## 关键文件
- plan: `submissions/TMLR/r3_plan.txt`(每天完成项即时打 ✅)
- 决策: `submissions/TMLR/r3_block0_decisions.md`
- 审稿: `submissions/TMLR/r3_mock_review.txt`
- Overleaf 源: `Overleaf_TMLR/`(sections/0_abstract..9_ethical + appendix.tex, tables/, figures/, images/)
- 分析产物: `tier0/18_t{6,7,8,9,10,11}_*.md/.csv`、`tier0/18_t4_pathA/`、`tier0/18_t1_plaintext_control/`
- Fig3 重绘脚本: `plots/fig3_prompt_group_bars.py`(新增 --panel 模式)

## 环境
- 本地 python: `/home/bc3194/miniconda3/envs/raven_rag/bin/python`(conda activate 本地损坏，直接调绝对路径)
- HPC ssh: `bc3194@jubail.abudhabi.nyu.edu`(不稳定，重试；scp HPC 文件到 /tmp 编辑再回推)
- HPC 项目路径: `/scratch/bc3194/dlm-jailbreak-transfer/`(T1 plaintext 输出、multiseed 输出、HF cache 均在其下)
- Overleaf 验证: hpcalendar 已部署(见 T19)
