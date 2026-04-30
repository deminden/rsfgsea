# Algorithm Guide

This project focuses on fgsea-compatible preranked enrichment.

## Supported Execution Paths

There are three maintained paths:

1. `fgsea` wrapper mode
2. explicit simple mode
3. explicit multilevel mode

There is also one hybrid acceleration path:

4. hybrid GPU wrapper mode

## Wrapper Mode

Wrapper mode is the closest match to how users normally call R `fgsea`.

Behavior:

- compute observed ES for each pathway using fgsea-compatible scoring
- run simple-stage permutations
- estimate NES and an initial p-value
- refine only the pathways that benefit from multilevel estimation

If `nperm` is provided, wrapper mode becomes simple mode.

## Simple Mode

Simple mode uses a fixed number of permutations for every tested pathway.

Use it when:

- you want predictable runtime
- you do not need very small p-values
- you are comparing against fixed-permutation reference runs

Tradeoff:

- simple mode is easier to reason about
- but it can be inefficient or low-resolution for very small p-values

## Multilevel Mode

Multilevel mode uses adaptive estimation to resolve small p-values more efficiently than brute-force fixed permutations.

Use it when:

- you care about tail probabilities
- wrapper mode would likely refine many significant pathways anyway

Key parameter:

- `sampleSize`

This is the multilevel sampling parameter and should stay aligned across comparisons if you care about parity.

## rsfgsea-decor

`rsfgsea-decor` is a decorrelation-inspired, redundancy-aware variant of preranked GSEA. It downweights pathway genes that are redundant with other genes in the same pathway, based on expression-derived correlations.

Classic mode uses:

```text
w_i = |stat_i|^gseaParam
```

Decor mode uses:

```text
w_i_decor = |stat_i|^gseaParam / (1 + alpha * r_i)
```

The first cache implementation computes `r_i` from the genes in each pathway that are present in the expression matrix. For `positive_mean`:

```text
r_i = mean(max(corr(i, j), 0)) over pathway genes j != i
```

For `abs_mean`, the absolute correlation is averaged instead. Gene identifiers are matched verbatim between ranks, GMT, and expression rows.

The decor cache stores derived redundancy scores, not the full pairwise correlation matrix. The transparent TSV cache includes metadata such as GMT SHA256, expression SHA256, correlation method, redundancy method, expression matrix assumptions, and row counts. `alpha` is not part of cache compatibility because it is applied at runtime.

Decor p-values use a conditional redundancy-profile null. For each pathway, observed ES uses each hit gene's own redundancy score. Each simple-mode null sample draws a random gene set of the same size, sorts sampled hit positions by rank, and applies the same ordered redundancy profile from the observed pathway. This conditions on the pathway's redundancy burden/profile while testing ranked-position enrichment.

Limitations in the first implementation:

- decor is experimental
- only CPU fixed-permutation simple mode is supported
- multilevel and GPU decor are rejected
- Pearson expression correlation is implemented; Spearman is reserved
- decor does not perform covariance whitening and does not fully decorrelate expression
- decor does not implement cameraPR or CorrSEA

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
