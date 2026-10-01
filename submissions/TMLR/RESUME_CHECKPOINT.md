# TMLR R3 — RESUME CHECKPOINT (2026-09-17)

READ THIS FIRST after the session gap. Everything below is verified on-disk as of 2026-09-17.

属性/惯例：回复用户用中文；代码仓库(本 repo)由用户手动 commit，我绝不代 commit；只有 Overleaf_TMLR(嵌套 LaTeX 论文 repo)按惯例无条件 commit+push。论文编译 28 页，0 error / 0 undefined / 0 overfull。所有修改要遵守用户偏好：不用 em/en-dash(散文中用逗号/冒号)、美式拼写、.json 一律 indent=4、"revision 文档更新同步在 plan 里打 ✅"。

## HARD REQUIREMENTS (before touching anything)
1. NESTED REPO: Overleaf_TMLR 是嵌套 git repo(remote=git.overleaf.com)。任何 Overleaf 编辑前先 `git -C Overleaf_TMLR pull origin main`。编辑后：latexmk 验证 → `git add <具体文件>` → `git -c user.email="bc3194@nyu.edu" -c user.name="Boyuan Chen" commit` → `git push origin main` → 报告 SHA 范围。不要 `git add -A`(会带入构建产物，除非确认 .gitignore 已覆盖)。
2. latexmk 验证用：`cd Overleaf_TMLR && latexmk -pdf -interaction=nonstopmode main.tex`，检查 0 undefined / 0 overfull / 0 error。若跑 `latexmk -c` 会清掉 .bbl 造成假 undefined + 损坏 pdf——修复 = `rm -f main.aux main.bbl main.blg main.log main.out main.pdf && pdflatex && bibtex && pdflatex && pdflatex`(完整三遍)。
3. 主 repo(本目录)从不 commit。工作树里未跟踪/已修改的内容提交与否交给用户。

## 项目状态（会话目标：TMLR R3 整稿修订，基于 r3_mock_review）

### Overleaf 已推送提交(最新在前，到 HEAD 6444ff0)
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
已完成打勾: BLOCK0(D1-D4) / BLOCK1(T1✅,T4✅;T2,T3⏳GPU) / BLOCK2(T6部分,T7全,T8全,T9全,T10全,T11全) / BLOCK3(T12-T15✅,T16分析✅正文insert依赖T5,T17✅) / BLOCK4(T18,T19,T20,T21,T22✅)。
DiffuCoder 碎片输出 = 稳定真实模型行为(plaintext 300 条也碎片化;3 seed 各 600 条统计完全一致 empty=85)。数据无需重跑。

## 下一步(阻断依赖：DECISION → 用户)
1. [ ] T5 人工标注(USER = 标注者)——最大阻塞。状态：230 行中已标 **46**(0×33/1×13, overlap 14/50 已填)，剩 **184 行**。
   标注文件：`tier0/18_t5_human_annotation/t5_annotation_sample_demasked.csv`(响应已 demask 还原成关键词，46 个已填标签保留)。规则(annotation_instruction 参照 memory)：1=harmful compliance，0=非。overlap=1 的 50 行是第二标注子集用于 kappa。
   解锁：T6 最终 precision-by-zone 措辞 + T16 混淆矩阵正文(HPC 混淆矩阵已对 GPT-5.6-sol 基准做完，待 human 标签替换参照)。
2. [ ] T23 GPU-hours 具体数字 + 硬件(审稿 M7)——正文现在写 "a few hundred GPU-hours"，需用户给真实数。
3. [ ] T24 vendor 披露状态/日期、T25 访问控制响应字段管理员姓名、T26 COI 申报(MetaCipher 作者重合)→ 纯事实，需用户。
4. [ ] BLOCK1 计算项 T2(LLaDA step-budget 匹配 bracket)、T3(llama multiseed)——若用户届时想跑 GPU。

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