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
