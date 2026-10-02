# T10 (W9) — Reliability evidence swap  [2026-09-15]

Data: multiseed replication (tier0/12 + tier0/14; paper tab:multiseed, 3-seed
DeepSeek binary temp 0): corrected diffusion seed-to-seed spread 0.0-2.0pp, max
range 2.0pp (LLaDA ArrAttack), all other cells <=1.0pp. Reproduction audit
(tier0/04): class-conditional concordance 28/51 = 54.9%; exact same-audit-judge
agreement is a separate 21/51 (41.2%) measure. Decomposition: 7 judge-boundary
cases and 9 regeneration-variance cases (n=16).

## Actions (T10 sub-bullets)
1. PROMOTE the three-seed replication (0.0-2.0pp spread) to HEADLINE reliability
   evidence for RQ2 / the diffusion cells. [MANUSCRIPT]
2. DEMOTE the 51-row reproduction audit to a spot-check-consistency check, not a
   fabrication test. Define 54.9% as class-conditional concordance, distinct from
   exact same-audit-judge agreement, and state that the 16-case decomposition is
   under-powered (n=16). [MANUSCRIPT]
3. CAVEAT (attach to the promoted seed evidence): the seed replication uses ONE
   common binary DeepSeek judge at temp 0, so flatness across seeds is
   judge-consistency + generation stability, not proof of either alone. [MANUSCRIPT]

Analysis is final (data referenced above already on disk); the three edits are
wording changes in the manuscript (Block 3).
