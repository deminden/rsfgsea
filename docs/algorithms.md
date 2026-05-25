# Algorithm Guide

This project exposes three user-facing GSEA tracks: decor first, classic fgsea-compatible workflows second, and native blitz third.

## Supported Execution Paths

There are five maintained CPU paths:

1. `rsfgsea-decor`
2. classic `fgsea` wrapper mode
3. classic explicit simple mode
4. classic explicit multilevel mode
5. native blitz mode

There is also one hybrid acceleration path:

6. hybrid GPU wrapper mode

## rsfgsea-decor

`rsfgsea-decor` is a decorrelation-inspired, redundancy-aware variant of preranked GSEA. It downweights pathway genes that are redundant with other genes in the same pathway, based on expression-derived correlations.

Classic mode uses:

```text
w_i = |stat_i|^gseaParam
```

The sensitive decor preset uses the `raw-rational` formula:

```text
w_i_decor = |stat_i|^gseaParam / (1 + alpha * r_i)
```

The first cache implementation computes `r_i` from the genes in each pathway that are present in the expression matrix. For `positive_mean`:

```text
r_i = mean(max(corr(i, j), 0)) over pathway genes j != i
```

For `abs_mean`, the absolute correlation is averaged instead. Gene identifiers are matched verbatim between ranks, GMT, and expression rows.

The decor cache stores derived redundancy scores, not the full pairwise correlation matrix. The transparent TSV cache includes metadata such as GMT SHA256, expression SHA256, correlation method, redundancy method, expression matrix assumptions, and row counts. `alpha` is not part of cache compatibility because it is applied at runtime.

For each pathway, observed decor ES uses each hit gene's own redundancy score. Each simple-mode permutation draws a random gene set of the same size, sorts sampled hit positions by rank, and applies the same ordered redundancy profile from the observed pathway. This keeps the pathway's redundancy burden fixed while testing ranked-position enrichment. Decor multilevel mode reuses that fixed-profile simple screen, then adaptively refines only pathways where the simple tail estimate is noisier than the decor multilevel estimate.

The base hit weight is always:

```text
b_i = |stat_i|^gseaParam
```

The selected preset converts raw redundancy `r_i` to a non-negative penalty:

- `raw-rational`: `1 / (1 + alpha * r)`
- `exp-scaled`: `exp(-alpha * r_scaled)` with median scaling
- `threshold-rational`: `1 / (1 + alpha * max(0, r - tau))`

The public presets are:

- `sensitive`: `raw-rational`, `alpha=22`
- `balanced`: `threshold-rational`, `tau=0.04`, `alpha=60`
- `specific`: `threshold-rational`, `tau=0.05`, `alpha=65`
- `strict`: `exp-scaled`, target median penalty `0.10`

For users who want a single high-level knob, `decor-stringency` maps onto this same calibrated preset ladder: `0 <= x < 35` selects `sensitive`, `35 <= x < 65` selects `balanced`, `65 <= x < 85` selects `specific`, and `85 <= x <= 100` selects `strict`. It autoswitches formula families by preset rather than interpolating unvalidated formula parameters.

Because GSEA hit increments are normalized by total hit weight, a uniform multiplicative penalty across all hits in one pathway cancels out in the observed ES walk. Presets mainly affect ES through relative differences among genes within a pathway. Pathway-wide redundancy burden is handled by the decor permutation calibration rather than by expecting uniform scaling to change observed ES.

Current limitations:

- decor is CPU-only; GPU decor is rejected
- decor multilevel is a decor-specific adaptive tail estimator rather than an R `fgseaMultilevel` parity target
- Pearson expression correlation is implemented; Spearman is reserved
- decor does not perform covariance whitening and does not fully decorrelate expression
- decor does not implement cameraPR or CorrSEA

## Classic fgsea-Compatible Modes

### Wrapper Mode

Wrapper mode is the closest match to how users normally call R `fgsea`.

Behavior:

- compute observed ES for each pathway using fgsea-compatible scoring
- run simple-stage permutations
- estimate NES and an initial p-value
- refine only the pathways that benefit from multilevel estimation

If `nperm` is provided, wrapper mode becomes simple mode.

### Simple Mode

Simple mode uses a fixed number of permutations for every tested pathway.

Use it when:

- you want predictable runtime
- you do not need very small p-values
- you are comparing against fixed-permutation reference runs

Tradeoff:

- simple mode is easier to reason about
- but it can be inefficient or low-resolution for very small p-values

### Multilevel Mode

Multilevel mode uses adaptive estimation to resolve small p-values more efficiently than brute-force fixed permutations.

Use it when:

- you care about tail probabilities
- wrapper mode would likely refine many significant pathways anyway

Key parameter:

- `sampleSize`

This is the multilevel sampling parameter and should stay aligned across comparisons if you care about parity.

## Blitz Mode

Blitz mode is a native Rust implementation of the `blitzgsea.gsea()` preranked workflow. It has its own mode-aware defaults: `permutations=1000`, `anchors=40`, `min_size=5`, `max_size=4000`, `processes=4`, `symmetric=false`, `seed=0`, `center=true`, `accuracy=40`, and `deep_accuracy=50`.

The implementation follows blitz preprocessing:

- sort the signature by descending score
- keep the first duplicate gene after sorting
- optionally center signature scores
- intersect pathway genes with the signature universe
- score with blitz leading-edge semantics

Blitz mode is intentionally separate from decor and classic fgsea-compatible modes. Library callers reuse native null-model fits for repeated identical calls through an in-process signature cache; one-shot CLI runs leave that cache off unless explicitly requested. It rejects `gpu`, decor mode, fixed `nperm`, non-`std` score types, and `gseaParam` values other than `1.0`. Result columns keep the rsfgsea shape: `pval` is the blitz p-value, `padj` is BH/FDR, and `log2err` is missing.

## Score Types

`std`

- two-sided selection
- chooses the larger absolute deviation

`pos`

- positive enrichment only

`neg`

- negative enrichment only

## ES, NES, And `log2err`

`ES`

- observed enrichment score for the pathway

`NES`

- normalized enrichment score using same-tail null means

`log2err`

- uncertainty estimate for the p-value
- most relevant in multilevel-refined outputs

## Hybrid GPU Path

The GPU implementation is intentionally hybrid.

What runs on GPU:

- null generation for simple-stage screening
- batched ES screening

What stays on CPU:

- parity-sensitive multilevel refinement
- final statistical decisions that depend on the CPU multilevel kernel

This design keeps the expensive wide screening work on GPU without replacing the parity-focused multilevel implementation.

## Determinism And Parity

The project aims at fgsea-compatible behavior, not a different statistical interpretation.

The important consequences are:

- RNG behavior matters
- score preparation matters
- wrapper/simple/multilevel routing matters
- thread-count changes should not change the intended statistical path

If you change any of those, parity tests are the first thing to revisit.

Current compatibility note:

- upstream `fgsea` has a user-visible single-pathway simple-stage RNG quirk
- `rsfgsea` currently preserves that behavior for strict parity with the
  released upstream package
- if upstream `fgsea` fixes the quirk, `rsfgsea` should follow that fix rather
  than preserving the old branch-specific RNG behavior indefinitely
