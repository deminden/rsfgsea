use crate::core::{BlitzOptions, EnrichmentResult, Pathway, RankedList};
use anyhow::{Result, bail};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::{OnceLock, RwLock};
use std::time::Instant;

mod compat;
mod smoothing;

use self::compat::{
    NumpyMt19937, PythonIntSet, PythonStringSet, numpy_hit_score_sum_f64, numpy_log_f32,
    numpy_pairwise_sum_f32, numpy_pairwise_sum_f64,
};
#[cfg(test)]
use self::compat::{python_ascii_hash_seed0, python_int_set_iteration_order};
#[cfg(test)]
use self::smoothing::lowess;
use self::smoothing::{LinearInterp, lowess_interpolation};

#[derive(Clone)]
struct BlitzSignature {
    genes: Vec<String>,
    abs_scores: Vec<f64>,
    gene_to_idx: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct GammaFit {
    alpha: f64,
    beta: f64,
}

#[derive(Debug, Clone)]
struct AnchorFit {
    alpha_pos: f64,
    beta_pos: f64,
    alpha_neg: f64,
    beta_neg: f64,
    pos_ratio: f64,
}

#[derive(Clone)]
struct BlitzModel {
    alpha_pos: LinearInterp,
    beta_pos: LinearInterp,
    pos_ratio: LinearInterp,
    alpha_neg: LinearInterp,
    beta_neg: LinearInterp,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BlitzModelCacheKey {
    signature_digest: [u8; 32],
    signature_len: usize,
    max_library_size: usize,
    permutations: usize,
    anchors: usize,
    processes: usize,
    symmetric: bool,
    seed: u64,
    center: bool,
}

static BLITZ_MODEL_CACHE: OnceLock<RwLock<HashMap<BlitzModelCacheKey, BlitzModel>>> =
    OnceLock::new();

#[derive(Clone)]
struct CleanedPathway {
    name: String,
    genes: Vec<String>,
    hit_indices: Vec<usize>,
    sorted_hit_indices: Vec<usize>,
    leading_hits: PythonIntSet,
}

#[derive(Clone, Copy)]
struct BlitzModelParams {
    pos_alpha: f64,
    pos_beta: f64,
    pos_ratio: f64,
    neg_alpha: f64,
    neg_beta: f64,
}

struct ScoredBlitzPathway {
    result: EnrichmentResult,
    cdf_calls: usize,
    fallback_calls: usize,
}

struct BlitzScoreScratch {
    hit_scores: Vec<f64>,
    leading_set: PythonIntSet,
}

impl BlitzScoreScratch {
    fn new(_signature_len: usize) -> Self {
        Self {
            hit_scores: Vec::new(),
            leading_set: PythonIntSet::new(),
        }
    }
}

struct GammaFitScratch {
    clean: Vec<f64>,
    clean_f32: Vec<f32>,
    log_f32: Vec<f32>,
}

impl GammaFitScratch {
    fn new(capacity: usize) -> Self {
        Self {
            clean: Vec::with_capacity(capacity),
            clean_f32: Vec::with_capacity(capacity),
            log_f32: Vec::with_capacity(capacity),
        }
    }
}

struct AnchorScratch {
    es: Vec<f64>,
    pos: Vec<f64>,
    neg: Vec<f64>,
    abs_samples: Vec<f64>,
    neg_abs: Vec<f64>,
    permutation: Vec<usize>,
    permutation_u32: Vec<u32>,
    sorted_hits: Vec<usize>,
    sorted_hits_u32: Vec<u32>,
    gamma_fit: GammaFitScratch,
}

impl AnchorScratch {
    fn new(signature_len: usize, permutations: usize) -> Self {
        Self {
            es: Vec::with_capacity(permutations),
            pos: Vec::with_capacity(permutations),
            neg: Vec::with_capacity(permutations),
            abs_samples: Vec::with_capacity(permutations),
            neg_abs: Vec::with_capacity(permutations),
            permutation: Vec::with_capacity(signature_len),
            permutation_u32: Vec::with_capacity(signature_len),
            sorted_hits: Vec::new(),
            sorted_hits_u32: Vec::new(),
            gamma_fit: GammaFitScratch::new(permutations),
        }
    }
}

struct EnrichmentExtrema {
    es: f64,
    peak_idx: usize,
    rmax: usize,
    max_value: f64,
    rmin: usize,
    min_value: f64,
}

struct BlitzTimingLogger {
    enabled: bool,
}

impl BlitzTimingLogger {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("RSFGSEA_BLITZ_TIMINGS").is_some(),
        }
    }

    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn finish(&self, stage: &str, start: Option<Instant>) {
        if let Some(start) = start {
            eprintln!(
                "RSFGSEA_BLITZ_TIMING\t{stage}\t{:.6}",
                start.elapsed().as_secs_f64()
            );
        }
    }

    fn value(&self, stage: &str, value: usize) {
        if self.enabled {
            eprintln!("RSFGSEA_BLITZ_TIMING\t{stage}\t{value}");
        }
    }
}

fn blitz_model_cache() -> &'static RwLock<HashMap<BlitzModelCacheKey, BlitzModel>> {
    BLITZ_MODEL_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

#[cfg(test)]
fn clear_blitz_model_cache() {
    if let Some(cache) = BLITZ_MODEL_CACHE.get() {
        cache
            .write()
            .expect("blitz model cache lock should not be poisoned")
            .clear();
    }
}

#[cfg(test)]
#[derive(Clone)]
struct HitMarker {
    marks: Vec<u32>,
    generation: u32,
}

#[cfg(test)]
impl HitMarker {
    fn new(len: usize) -> Self {
        Self {
            marks: vec![0; len],
            generation: 1,
        }
    }

    fn mark_hits(&mut self, hits: &[usize]) {
        self.next_generation();
        for &hit in hits {
            self.marks[hit] = self.generation;
        }
    }

    fn contains(&self, idx: usize) -> bool {
        self.marks[idx] == self.generation
    }

    fn next_generation(&mut self) {
        if self.generation == u32::MAX {
            self.marks.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }
}

pub fn fgsea_blitz_with_options(
    ranks: &RankedList,
    pathways: &[Pathway],
    options: &BlitzOptions,
) -> Result<Vec<EnrichmentResult>> {
    validate_options(options, ranks.len())?;

    let timings = BlitzTimingLogger::new();
    let stage = timings.start();
    let signature = prepare_signature(ranks, options.center);
    timings.finish("prepare_signature", stage);

    let stage = timings.start();
    let cleaned = clean_pathways(pathways, &signature);
    timings.finish("clean_pathways", stage);
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }

    // Calibrate one library-level null model, then score each pathway with the
    // interpolated parameters for its observed size.
    let stage = timings.start();
    let (model, model_cache_hit) = estimate_model_with_cache(&signature, &cleaned, options)?;
    timings.finish("estimate_model", stage);
    timings.value("model_cache_hit", usize::from(model_cache_hit));

    let stage = timings.start();
    let params_by_size = model_params_by_size(&model, &cleaned, options);
    timings.finish("model_param_cache", stage);

    let stage = timings.start();
    let scored = score_blitz_pathways(&signature, &cleaned, &params_by_size, options)?;
    timings.finish("final_es_leading_edge_gamma", stage);
    timings.value(
        "gamma_cdf_calls",
        scored.iter().map(|row| row.cdf_calls).sum::<usize>(),
    );
    timings.value(
        "gamma_fallback_calls",
        scored.iter().map(|row| row.fallback_calls).sum::<usize>(),
    );

    let mut results = scored.into_iter().map(|row| row.result).collect::<Vec<_>>();

    let stage = timings.start();
    apply_statsmodels_bh_adjustment(&mut results);
    results.sort_by(|a, b| {
        a.p_value
            .partial_cmp(&b.p_value)
            .unwrap()
            .then_with(|| a.pathway_name.cmp(&b.pathway_name))
    });
    timings.finish("bh_and_sort", stage);
    Ok(results)
}

#[doc(hidden)]
pub struct BlitzBenchPrepared {
    signature: BlitzSignature,
    cleaned: Vec<CleanedPathway>,
    params_by_size: Vec<Option<BlitzModelParams>>,
    options: BlitzOptions,
}

#[doc(hidden)]
pub fn __bench_blitz_prepare_scoring(
    ranks: &RankedList,
    pathways: &[Pathway],
    options: &BlitzOptions,
) -> Result<BlitzBenchPrepared> {
    validate_options(options, ranks.len())?;
    let signature = prepare_signature(ranks, options.center);
    let cleaned = clean_pathways(pathways, &signature);
    let model = estimate_model(&signature, &cleaned, options)?;
    let params_by_size = model_params_by_size(&model, &cleaned, options);
    Ok(BlitzBenchPrepared {
        signature,
        cleaned,
        params_by_size,
        options: options.clone(),
    })
}

#[doc(hidden)]
pub fn __bench_blitz_anchor_calibration(
    ranks: &RankedList,
    pathways: &[Pathway],
    options: &BlitzOptions,
) -> Result<usize> {
    validate_options(options, ranks.len())?;
    let signature = prepare_signature(ranks, options.center);
    let cleaned = clean_pathways(pathways, &signature);
    let (_, fits) = estimate_anchor_fits(&signature, &cleaned, options)?;
    Ok(fits.len())
}

#[doc(hidden)]
pub fn __bench_blitz_score_prepared(prepared: &BlitzBenchPrepared) -> Result<usize> {
    let scored = score_blitz_pathways(
        &prepared.signature,
        &prepared.cleaned,
        &prepared.params_by_size,
        &prepared.options,
    )?;
    Ok(scored.len())
}

#[doc(hidden)]
pub fn __bench_blitz_tail_microcases(deep_accuracy: usize) -> Result<f64> {
    let cases = [
        (
            crate::blitz_mpmath::TailBranch::Positive,
            0.73,
            2.8,
            0.19,
            0.57,
        ),
        (
            crate::blitz_mpmath::TailBranch::Negative,
            1.19,
            4.2,
            0.31,
            0.44,
        ),
        (
            crate::blitz_mpmath::TailBranch::Positive,
            13.0,
            1.4,
            0.18,
            0.63,
        ),
        (
            crate::blitz_mpmath::TailBranch::Negative,
            1.0e-12,
            3.5,
            0.42,
            0.51,
        ),
    ];
    let mut sum = 0.0;
    for (branch, x, alpha, beta, pos_ratio) in cases {
        let (tail, _) = gamma_tail_probability_with_fallback_flag(
            branch,
            x,
            alpha,
            beta,
            pos_ratio,
            deep_accuracy,
        )?;
        sum += tail.p_value + tail.prob_two_tailed + tail.gamma_prob;
    }
    Ok(sum)
}

fn model_params_by_size(
    model: &BlitzModel,
    cleaned: &[CleanedPathway],
    options: &BlitzOptions,
) -> Vec<Option<BlitzModelParams>> {
    let max_size = cleaned
        .iter()
        .map(|pathway| pathway.genes.len())
        .max()
        .unwrap_or(0)
        .min(options.max_size);
    let mut params_by_size = vec![None; max_size + 1];
    for pathway in cleaned {
        let size = pathway.genes.len();
        if size < options.min_size || size > options.max_size {
            continue;
        }
        if size >= params_by_size.len() {
            continue;
        }
        if params_by_size[size].is_none() {
            let size_f = size as f64;
            params_by_size[size] = Some(BlitzModelParams {
                pos_alpha: model.alpha_pos.at(size_f),
                pos_beta: model.beta_pos.at(size_f),
                pos_ratio: model.pos_ratio.at(size_f).clamp(0.0, 1.0),
                neg_alpha: model.alpha_neg.at(size_f),
                neg_beta: model.beta_neg.at(size_f),
            });
        }
    }
    params_by_size
}

fn score_blitz_pathways(
    signature: &BlitzSignature,
    cleaned: &[CleanedPathway],
    params_by_size: &[Option<BlitzModelParams>],
    options: &BlitzOptions,
) -> Result<Vec<ScoredBlitzPathway>> {
    let jobs = cleaned
        .iter()
        .filter_map(|pathway| {
            let size = pathway.genes.len();
            if size < options.min_size || size > options.max_size {
                return None;
            }
            let params = params_by_size
                .get(size)
                .and_then(|params| *params)
                .expect("model parameters should be cached for every kept size");
            Some((pathway, params))
        })
        .collect::<Vec<_>>();
    let processes = options.processes.max(1);
    let score = |scratch: &mut BlitzScoreScratch,
                 (pathway, params): &(&CleanedPathway, BlitzModelParams)|
     -> Result<ScoredBlitzPathway> {
        score_cleaned_pathway(signature, pathway, *params, options.deep_accuracy, scratch)
    };

    if processes == 1 {
        let mut scratch = BlitzScoreScratch::new(signature.abs_scores.len());
        let mut out = Vec::with_capacity(jobs.len());
        for job in &jobs {
            out.push(score(&mut scratch, job)?);
        }
        return Ok(out);
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()?;
    pool.install(|| {
        let min_len = 128.min(jobs.len().max(1));
        jobs.par_iter()
            .with_min_len(min_len)
            .map_init(
                || BlitzScoreScratch::new(signature.abs_scores.len()),
                |scratch, job| score(scratch, job),
            )
            .collect::<Result<Vec<_>>>()
    })
}

fn score_cleaned_pathway(
    signature: &BlitzSignature,
    pathway: &CleanedPathway,
    params: BlitzModelParams,
    deep_accuracy: usize,
    scratch: &mut BlitzScoreScratch,
) -> Result<ScoredBlitzPathway> {
    let extrema = enrichment_score_for_indices(
        &signature.abs_scores,
        &pathway.hit_indices,
        &pathway.sorted_hit_indices,
        scratch,
    );
    let leading_edge = leading_edge_blitz(
        signature,
        &pathway.leading_hits,
        extrema.rmax,
        extrema.max_value,
        extrema.rmin,
        extrema.min_value,
        extrema.peak_idx,
        scratch,
    );
    let es = extrema.es;
    let (p_value, nes, fallback_used) = if es > 0.0 {
        let (tail, fallback_used) = gamma_tail_probability_with_fallback_flag(
            crate::blitz_mpmath::TailBranch::Positive,
            es,
            params.pos_alpha,
            params.pos_beta,
            params.pos_ratio,
            deep_accuracy,
        )?;
        let nes = normal_isf(tail.prob_two_tailed);
        (tail.p_value, Some(nes), fallback_used)
    } else {
        let (tail, fallback_used) = gamma_tail_probability_with_fallback_flag(
            crate::blitz_mpmath::TailBranch::Negative,
            -es,
            params.neg_alpha,
            params.neg_beta,
            params.pos_ratio,
            deep_accuracy,
        )?;
        let mut nes = -normal_isf(tail.prob_two_tailed);
        if nes == 0.0 {
            nes = -0.0;
        }
        (tail.p_value, Some(nes), fallback_used)
    };

    Ok(ScoredBlitzPathway {
        result: EnrichmentResult {
            pathway_name: pathway.name.clone(),
            size: pathway.genes.len(),
            es,
            nes,
            p_value,
            padj: None,
            log2err: None,
            leading_edge,
        },
        cdf_calls: 1,
        fallback_calls: usize::from(fallback_used),
    })
}

fn validate_options(options: &BlitzOptions, n_genes: usize) -> Result<()> {
    if n_genes == 0 {
        bail!("blitz mode requires at least one ranked gene.");
    }
    if options.permutations == 0 {
        bail!("blitz permutations must be greater than 0.");
    }
    if options.anchors == 0 {
        bail!("blitz anchors must be greater than 0.");
    }
    if options.min_size == 0 {
        bail!("blitz minSize must be greater than 0.");
    }
    if options.max_size == 0 {
        bail!("blitz maxSize must be greater than 0.");
    }
    if options.processes == 0 {
        bail!("blitz processes must be greater than 0.");
    }
    Ok(())
}

fn prepare_signature(ranks: &RankedList, center: bool) -> BlitzSignature {
    let mut rows = ranks
        .genes
        .iter()
        .cloned()
        .zip(ranks.scores.iter().copied())
        .enumerate()
        .map(|(idx, (gene, score))| (idx, gene, score))
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut genes = Vec::new();
    let mut scores = Vec::new();
    let mut seen = HashSet::new();
    for (_, gene, score) in rows {
        if seen.insert(gene.clone()) {
            genes.push(gene);
            scores.push(score);
        }
    }
    if center && !scores.is_empty() {
        let mean = numpy_pairwise_sum_f64(&scores) / scores.len() as f64;
        for score in &mut scores {
            *score -= mean;
        }
    }
    let abs_scores = scores.iter().map(|v| v.abs()).collect::<Vec<_>>();
    let gene_to_idx = genes
        .iter()
        .enumerate()
        .map(|(i, gene)| (gene.clone(), i))
        .collect();
    BlitzSignature {
        genes,
        abs_scores,
        gene_to_idx,
    }
}

fn clean_pathways(pathways: &[Pathway], signature: &BlitzSignature) -> Vec<CleanedPathway> {
    let signature_genes = signature.genes.iter().cloned().collect::<HashSet<_>>();
    let signature_set_order = PythonStringSet::from_iter(signature.genes.iter().cloned());
    pathways
        .iter()
        .map(|pathway| {
            // Match CPython set intersection order so leading-edge output stays
            // bit-for-bit comparable with the blitz reference implementation.
            let gene_set = PythonStringSet::from_iter(pathway.genes.iter().cloned());
            let genes = if gene_set.len() <= signature_genes.len() {
                gene_set.intersection_new_set_order(&signature_genes)
            } else {
                signature_set_order.intersection_new_set_order(&gene_set.members())
            };
            let hit_indices = genes
                .iter()
                .filter_map(|gene| signature.gene_to_idx.get(gene).copied())
                .collect::<Vec<_>>();
            let mut sorted_hit_indices = hit_indices.clone();
            sorted_hit_indices.sort_unstable();
            let leading_gene_set = PythonStringSet::from_iter(genes.iter().cloned());
            let leading_hits = PythonIntSet::from_iter(
                leading_gene_set
                    .iter_values()
                    .filter_map(|gene| signature.gene_to_idx.get(gene).copied()),
            );
            CleanedPathway {
                name: pathway.name.clone(),
                genes,
                hit_indices,
                sorted_hit_indices,
                leading_hits,
            }
        })
        .collect()
}

fn max_cleaned_library_size(library: &[CleanedPathway]) -> usize {
    library
        .iter()
        .map(|pathway| pathway.genes.len())
        .max()
        .unwrap_or(1)
        .max(1)
}

fn finalize_sha256(hasher: Sha256) -> [u8; 32] {
    let digest = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn update_usize_digest(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn blitz_signature_digest(signature: &BlitzSignature) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_usize_digest(&mut hasher, signature.genes.len());
    for score in &signature.abs_scores {
        hasher.update(score.to_bits().to_le_bytes());
    }
    finalize_sha256(hasher)
}

fn blitz_model_cache_key(
    signature: &BlitzSignature,
    library: &[CleanedPathway],
    options: &BlitzOptions,
) -> BlitzModelCacheKey {
    BlitzModelCacheKey {
        signature_digest: blitz_signature_digest(signature),
        signature_len: signature.genes.len(),
        max_library_size: max_cleaned_library_size(library),
        permutations: options.permutations,
        anchors: options.anchors,
        processes: options.processes,
        symmetric: options.symmetric,
        seed: options.seed,
        center: options.center,
    }
}

fn apply_statsmodels_bh_adjustment(results: &mut [EnrichmentResult]) {
    if results.is_empty() {
        return;
    }
    let mut indices = (0..results.len())
        .filter(|&idx| results[idx].p_value.is_finite())
        .collect::<Vec<_>>();
    indices.sort_by(|&a, &b| results[a].p_value.partial_cmp(&results[b].p_value).unwrap());
    let m = indices.len() as f64;
    let mut prev = 1.0;
    for rank_idx in (0..indices.len()).rev() {
        let idx = indices[rank_idx];
        let ecdf = (rank_idx + 1) as f64 / m;
        let corrected = (results[idx].p_value / ecdf).min(prev).min(1.0);
        results[idx].padj = Some(corrected);
        prev = corrected;
    }
}

fn estimate_model(
    signature: &BlitzSignature,
    library: &[CleanedPathway],
    options: &BlitzOptions,
) -> Result<BlitzModel> {
    let (anchor_sizes, fits) = estimate_anchor_fits(signature, library, options)?;

    let x = anchor_sizes.iter().map(|&v| v as f64).collect::<Vec<_>>();
    let alpha_pos = fits.iter().map(|fit| fit.alpha_pos).collect::<Vec<_>>();
    let beta_pos = fits.iter().map(|fit| fit.beta_pos).collect::<Vec<_>>();
    let alpha_neg = fits.iter().map(|fit| fit.alpha_neg).collect::<Vec<_>>();
    let beta_neg = fits.iter().map(|fit| fit.beta_neg).collect::<Vec<_>>();
    let mut jitter_rng = NumpyMt19937::new(options.seed as u32);
    let jitters = jitter_rng.standard_normals(fits.len());
    // Upstream blitz applies a tiny one-sided jitter before smoothing the
    // positive-tail mixture ratio; keep that quirk isolated here.
    let pos_ratio = fits
        .iter()
        .zip(jitters)
        .map(|(fit, jitter)| (fit.pos_ratio - (0.0001 * jitter).abs()).clamp(0.0, 1.0))
        .collect::<Vec<_>>();

    Ok(BlitzModel {
        alpha_pos: lowess_interpolation(&x, &alpha_pos, 0.6),
        beta_pos: lowess_interpolation(&x, &beta_pos, 0.15),
        pos_ratio: lowess_interpolation(&x, &pos_ratio, 0.5),
        alpha_neg: lowess_interpolation(&x, &alpha_neg, 0.6),
        beta_neg: lowess_interpolation(&x, &beta_neg, 0.15),
    })
}

fn estimate_model_with_cache(
    signature: &BlitzSignature,
    library: &[CleanedPathway],
    options: &BlitzOptions,
) -> Result<(BlitzModel, bool)> {
    if !options.signature_cache {
        return estimate_model(signature, library, options).map(|model| (model, false));
    }

    let key = blitz_model_cache_key(signature, library, options);
    if let Some(model) = blitz_model_cache()
        .read()
        .expect("blitz model cache lock should not be poisoned")
        .get(&key)
        .cloned()
    {
        return Ok((model, true));
    }

    let model = estimate_model(signature, library, options)?;
    let mut cache = blitz_model_cache()
        .write()
        .expect("blitz model cache lock should not be poisoned");
    if let Some(model) = cache.get(&key).cloned() {
        return Ok((model, true));
    }
    cache.insert(key, model.clone());
    Ok((model, false))
}

fn estimate_anchor_fits(
    signature: &BlitzSignature,
    library: &[CleanedPathway],
    options: &BlitzOptions,
) -> Result<(Vec<usize>, Vec<AnchorFit>)> {
    let max_library_size = max_cleaned_library_size(library);
    let anchor_sizes = anchor_set_sizes(max_library_size, signature.genes.len(), options.anchors);
    if anchor_sizes.is_empty() {
        bail!("blitz calibration produced no valid anchor set sizes.");
    }
    let processes = options.processes.max(1);
    let fits = if processes == 1 {
        let mut rng = NumpyMt19937::new(options.seed as u32);
        let mut scratch = AnchorScratch::new(signature.abs_scores.len(), options.permutations);
        let mut fits = Vec::with_capacity(anchor_sizes.len());
        for &size in &anchor_sizes {
            fits.push(estimate_anchor(
                signature,
                size,
                options,
                &mut rng,
                &mut scratch,
            )?);
        }
        fits
    } else {
        // Anchor jobs are partitioned deterministically because the reference
        // traces depend on exact RNG consumption, not just aggregate results.
        let worker_count = processes.min(anchor_sizes.len());
        let mut workers = vec![NumpyMt19937::new(options.seed as u32); worker_count];
        let mut per_worker: Vec<Vec<(usize, usize)>> = vec![Vec::new(); worker_count];
        for (i, &size) in anchor_sizes.iter().enumerate() {
            per_worker[reference_worker_index(i, worker_count)].push((i, size));
        }
        let mut out: Vec<Option<AnchorFit>> = vec![None; anchor_sizes.len()];
        let worker_results = per_worker
            .into_par_iter()
            .zip(workers.par_iter_mut())
            .map(|(jobs, rng)| {
                let mut scratch =
                    AnchorScratch::new(signature.abs_scores.len(), options.permutations);
                jobs.into_iter()
                    .map(|(idx, size)| {
                        estimate_anchor(signature, size, options, rng, &mut scratch)
                            .map(|fit| (idx, fit))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        for worker in worker_results {
            for (idx, fit) in worker {
                out[idx] = Some(fit);
            }
        }
        out.into_iter()
            .map(|fit| fit.expect("every anchor fit should be populated"))
            .collect()
    };
    Ok((anchor_sizes, fits))
}

fn reference_worker_index(task_idx: usize, worker_count: usize) -> usize {
    if worker_count == 4 {
        if task_idx < 4 {
            task_idx
        } else {
            [1, 2, 0, 3][(task_idx - 4) % 4]
        }
    } else {
        task_idx % worker_count
    }
}

fn anchor_set_sizes(max_library_size: usize, signature_len: usize, anchors: usize) -> Vec<usize> {
    let mut values = Vec::new();
    if anchors == 1 {
        values.push(1);
    } else {
        for i in 0..anchors {
            let value =
                1.0 + (max_library_size.saturating_sub(1)) as f64 * i as f64 / (anchors - 1) as f64;
            values.push(value as usize);
        }
    }
    values.extend([
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        12,
        16,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        100,
        max_library_size + 10,
        max_library_size + 30,
    ]);
    values.sort_unstable();
    values.dedup();
    values
        .into_iter()
        .filter(|&size| size > 0 && size < signature_len)
        .collect()
}

fn estimate_anchor(
    signature: &BlitzSignature,
    set_size: usize,
    options: &BlitzOptions,
    rng: &mut NumpyMt19937,
    scratch: &mut AnchorScratch,
) -> Result<AnchorFit> {
    scratch.es.clear();
    if let Ok(n_u32) = u32::try_from(signature.abs_scores.len()) {
        for _ in 0..options.permutations {
            let hits = rng.choice_without_replacement_u32_into(
                n_u32,
                set_size,
                &mut scratch.permutation_u32,
            );
            let value = enrichment_score_for_hits_u32(
                &signature.abs_scores,
                hits,
                &mut scratch.sorted_hits_u32,
            );
            if value.is_finite() {
                scratch.es.push(value);
            }
        }
    } else {
        for _ in 0..options.permutations {
            let hits = rng.choice_without_replacement_into(
                signature.abs_scores.len(),
                set_size,
                &mut scratch.permutation,
            );
            let value =
                enrichment_score_for_hits(&signature.abs_scores, hits, &mut scratch.sorted_hits);
            if value.is_finite() {
                scratch.es.push(value);
            }
        }
    }
    if scratch.es.is_empty() {
        bail!("blitz calibration generated no finite enrichment scores for set size {set_size}.");
    }
    scratch.pos.clear();
    scratch.neg.clear();
    for &value in &scratch.es {
        if value > 0.0 {
            scratch.pos.push(value);
        } else if value < 0.0 {
            scratch.neg.push(value);
        }
    }
    let mut symmetric = options.symmetric;
    if (scratch.neg.len() < 250 || scratch.pos.len() < 250) && !symmetric {
        // Very small one-sided samples make separate gamma fits noisy; the
        // symmetric fallback borrows both tails while preserving the sign ratio.
        symmetric = true;
    }

    let (pos_fit, neg_fit) = if symmetric {
        scratch.abs_samples.clear();
        scratch.abs_samples.extend(
            scratch
                .es
                .iter()
                .copied()
                .filter(|v| *v != 0.0)
                .map(f64::abs),
        );
        let fit = fit_gamma_floc0_with_scratch(&scratch.abs_samples, &mut scratch.gamma_fit)?;
        (fit.clone(), fit)
    } else {
        let pos_fit = fit_gamma_floc0_with_scratch(&scratch.pos, &mut scratch.gamma_fit)?;
        scratch.neg_abs.clear();
        scratch.neg_abs.extend(scratch.neg.iter().map(|v| -*v));
        let neg_fit = fit_gamma_floc0_with_scratch(&scratch.neg_abs, &mut scratch.gamma_fit)?;
        (pos_fit, neg_fit)
    };

    let denom = scratch.pos.len() + scratch.neg.len();
    let pos_ratio = if denom == 0 {
        0.5
    } else {
        scratch.pos.len() as f64 / denom as f64
    };

    Ok(AnchorFit {
        alpha_pos: pos_fit.alpha,
        beta_pos: pos_fit.beta,
        alpha_neg: neg_fit.alpha,
        beta_neg: neg_fit.beta,
        pos_ratio,
    })
}

fn fit_gamma_floc0_with_scratch(values: &[f64], scratch: &mut GammaFitScratch) -> Result<GammaFit> {
    scratch.clean.clear();
    scratch
        .clean
        .extend(values.iter().copied().filter(|v| v.is_finite() && *v > 0.0));
    if scratch.clean.is_empty() {
        bail!("cannot fit gamma distribution to an empty positive sample.");
    }
    scratch.clean_f32.clear();
    scratch
        .clean_f32
        .extend(scratch.clean.iter().map(|v| *v as f32));
    scratch.log_f32.clear();
    scratch
        .log_f32
        .extend(scratch.clean_f32.iter().map(|v| numpy_log_f32(*v)));
    let n = scratch.clean_f32.len() as f32;
    let mean = numpy_pairwise_sum_f32(&scratch.clean_f32) / n;
    let mean_log = numpy_pairwise_sum_f32(&scratch.log_f32) / n;
    let s = numpy_log_f32(mean) - mean_log;
    let estimate = (3.0_f32 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s);
    let alpha =
        solve_gamma_shape_brentq((estimate * 0.6) as f64, (estimate * 1.4) as f64, s as f64);
    Ok(GammaFit {
        alpha,
        beta: (mean / alpha as f32) as f64,
    })
}

fn solve_gamma_shape_brentq(lo: f64, hi: f64, s: f64) -> f64 {
    let f = |a: f64| a.ln() - scipy_digamma(a) - s;
    let mut fpre = f(lo);
    let mut fcur = f(hi);
    if fpre == 0.0 {
        return lo;
    }
    if fcur == 0.0 {
        return hi;
    }
    if !fpre.is_finite() || !fcur.is_finite() || fpre.is_sign_negative() == fcur.is_sign_negative()
    {
        return (lo + hi) * 0.5;
    }

    let mut xpre = lo;
    let mut xcur = hi;
    let mut xblk = 0.0;
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;
    const XTOL: f64 = 2e-12;
    const RTOL: f64 = 8.881_784_197_001_252e-16;

    for _ in 0..100 {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }

        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (XTOL + RTOL * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    xcur
}

fn scipy_digamma(x: f64) -> f64 {
    const NEG_ROOT: f64 = -0.504_083_008_264_455_4;
    const NEG_ROOT_VAL: f64 = 7.289_763_902_976_895e-17;

    if (x - NEG_ROOT).abs() < 0.3 {
        return scipy_digamma_zeta_series(x, NEG_ROOT, NEG_ROOT_VAL);
    }
    scipy_cephes_psi(x)
}

fn scipy_digamma_zeta_series(mut z: f64, root: f64, root_val: f64) -> f64 {
    let mut res = root_val;
    let mut coeff = -1.0;
    z -= root;
    for n in 1..100 {
        coeff *= -z;
        let term = coeff * scipy_hurwitz_zeta(n + 1, root);
        res += term;
        if term.abs() < f64::EPSILON * res.abs() {
            break;
        }
    }
    res
}

fn scipy_hurwitz_zeta(n: i32, q: f64) -> f64 {
    // Only used for the negative-root digamma series; blitz gamma fits stay positive.
    (0..200_000)
        .map(|k| (q + k as f64).powi(-n))
        .take_while(|term| term.is_finite() && term.abs() > f64::EPSILON)
        .sum()
}

fn scipy_cephes_psi(mut x: f64) -> f64 {
    const EULER: f64 = 0.577_215_664_901_532_9;
    let mut y = 0.0;

    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(-x);
    }
    if x < 0.0 {
        let r = x.fract();
        if r == 0.0 {
            return f64::NAN;
        }
        y = -std::f64::consts::PI / (std::f64::consts::PI * r).tan();
        x = 1.0 - x;
    }

    if x <= 10.0 && x == x.floor() {
        let n = x as i32;
        for i in 1..n {
            y += 1.0 / i as f64;
        }
        return y - EULER;
    }

    if x < 1.0 {
        y -= 1.0 / x;
        x += 1.0;
    } else if x < 10.0 {
        while x > 2.0 {
            x -= 1.0;
            y += 1.0 / x;
        }
    }
    if (1.0..=2.0).contains(&x) {
        return y + scipy_digamma_imp_1_2(x);
    }
    y + scipy_psi_asy(x)
}

#[allow(clippy::excessive_precision)]
fn scipy_digamma_imp_1_2(x: f64) -> f64 {
    const Y: f64 = 0.995_581_626_892_089_8;
    const ROOT1: f64 = 1_569_415_565.0 / 1_073_741_824.0;
    const ROOT2: f64 = (381_566_830.0 / 1_073_741_824.0) / 1_073_741_824.0;
    const ROOT3: f64 = 0.901_631_209_325_869_6e-19;
    const P: [f64; 6] = [
        -0.0020713321167745952,
        -0.045251321448739056,
        -0.28919126444774784,
        -0.65031853770896507,
        -0.32555031186804491,
        0.25479851061131551,
    ];
    const Q: [f64; 7] = [
        -0.55789841321675513e-6,
        0.0021284987017821144,
        0.054151797245674225,
        0.43593529692665969,
        1.4606242909763515,
        2.0767117023730469,
        1.0,
    ];
    let mut g = x - ROOT1;
    g -= ROOT2;
    g -= ROOT3;
    let r = scipy_polevl(x - 1.0, &P) / scipy_polevl(x - 1.0, &Q);
    g * Y + g * r
}

#[allow(clippy::excessive_precision)]
fn scipy_psi_asy(x: f64) -> f64 {
    const A: [f64; 7] = [
        8.33333333333333333333E-2,
        -2.10927960927960927961E-2,
        7.57575757575757575758E-3,
        -4.16666666666666666667E-3,
        3.96825396825396825397E-3,
        -8.33333333333333333333E-3,
        8.33333333333333333333E-2,
    ];
    let y = if x < 1.0e17 {
        let z = 1.0 / (x * x);
        z * scipy_polevl(z, &A)
    } else {
        0.0
    };
    x.ln() - 0.5 / x - y
}

fn scipy_polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for coeff in &coef[1..] {
        ans = ans * x + coeff;
    }
    ans
}

fn scipy_p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for coeff in &coef[1..] {
        ans = ans * x + coeff;
    }
    ans
}

#[allow(clippy::excessive_precision)]
fn scipy_cephes_sqrtpi() -> f64 {
    2.50662827463100050242E0
}

fn enrichment_score_for_indices(
    abs_scores: &[f64],
    hits: &[usize],
    sorted_hits: &[usize],
    scratch: &mut BlitzScoreScratch,
) -> EnrichmentExtrema {
    let number_hits = hits.len();
    let number_miss = abs_scores.len().saturating_sub(number_hits);
    scratch.hit_scores.clear();
    scratch
        .hit_scores
        .extend(hits.iter().map(|&idx| abs_scores[idx]));
    let sum_hit_scores = numpy_hit_score_sum_f64(&scratch.hit_scores);
    let norm_hit = if sum_hit_scores == 0.0 {
        0.0
    } else {
        1.0 / sum_hit_scores
    };
    let norm_no_hit = if number_miss == 0 {
        0.0
    } else {
        1.0 / number_miss as f64
    };

    let mut csum = 0.0;
    let mut best_idx = 0usize;
    let mut best_value = 0.0;
    let mut best_abs = f64::NEG_INFINITY;
    let mut rmax = 0usize;
    let mut rmin = 0usize;
    let mut max_value = f64::NEG_INFINITY;
    let mut min_value = f64::INFINITY;
    let mut rank = 0usize;
    for &hit in sorted_hits {
        if rank < hit {
            advance_miss_gap_f64(
                &mut csum,
                rank,
                hit - rank,
                -norm_no_hit,
                &mut best_idx,
                &mut best_value,
                &mut best_abs,
                &mut rmax,
                &mut max_value,
                &mut rmin,
                &mut min_value,
            );
        }
        csum += abs_scores[hit] * norm_hit;
        if csum >= max_value {
            max_value = csum;
            rmax = hit;
        }
        if csum <= min_value {
            min_value = csum;
            rmin = hit;
        }
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best_idx = hit;
            best_value = csum;
        }
        rank = hit + 1;
    }
    if rank < abs_scores.len() {
        advance_miss_gap_f64(
            &mut csum,
            rank,
            abs_scores.len() - rank,
            -norm_no_hit,
            &mut best_idx,
            &mut best_value,
            &mut best_abs,
            &mut rmax,
            &mut max_value,
            &mut rmin,
            &mut min_value,
        );
    }
    EnrichmentExtrema {
        es: if abs_scores.is_empty() {
            0.0
        } else {
            best_value
        },
        peak_idx: best_idx,
        rmax,
        max_value,
        rmin,
        min_value,
    }
}

fn enrichment_score_for_hits(
    abs_scores: &[f64],
    hits: &[usize],
    sorted_hits: &mut Vec<usize>,
) -> f64 {
    let number_hits = hits.len();
    let number_miss = abs_scores.len().saturating_sub(number_hits);
    sorted_hits.clear();
    let mut sum_hit_scores = 0.0;
    for &idx in hits {
        sum_hit_scores += abs_scores[idx];
        sorted_hits.push(idx);
    }
    if sum_hit_scores == 0.0 || number_miss == 0 {
        return 0.0;
    }
    let norm_hit = 1.0 / sum_hit_scores;
    let norm_no_hit = 1.0 / number_miss as f64;
    let mut csum = 0.0_f32;
    let mut best = 0.0_f32;
    let mut best_abs = f32::NEG_INFINITY;

    sorted_hits.sort_unstable();

    let mut rank = 0usize;
    for &hit in sorted_hits.iter() {
        if rank < hit {
            csum = advance_miss_gap_f32(
                csum,
                hit - rank,
                (-norm_no_hit) as f32,
                &mut best,
                &mut best_abs,
            );
        }
        let increment = 1.0 * (abs_scores[hit] * norm_hit + norm_no_hit) - norm_no_hit;
        csum += increment as f32;
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best = csum;
        }
        rank = hit + 1;
    }
    if rank < abs_scores.len() {
        advance_miss_gap_f32(
            csum,
            abs_scores.len() - rank,
            (-norm_no_hit) as f32,
            &mut best,
            &mut best_abs,
        );
    }
    best as f64
}

fn enrichment_score_for_hits_u32(
    abs_scores: &[f64],
    hits: &[u32],
    sorted_hits: &mut Vec<u32>,
) -> f64 {
    let number_hits = hits.len();
    let number_miss = abs_scores.len().saturating_sub(number_hits);
    sorted_hits.clear();
    let mut sum_hit_scores = 0.0;
    for &idx in hits {
        sum_hit_scores += abs_scores[idx as usize];
        sorted_hits.push(idx);
    }
    if sum_hit_scores == 0.0 || number_miss == 0 {
        return 0.0;
    }
    let norm_hit = 1.0 / sum_hit_scores;
    let norm_no_hit = 1.0 / number_miss as f64;
    let mut csum = 0.0_f32;
    let mut best = 0.0_f32;
    let mut best_abs = f32::NEG_INFINITY;

    sorted_hits.sort_unstable();

    let mut rank = 0usize;
    for &hit in sorted_hits.iter() {
        let hit = hit as usize;
        if rank < hit {
            csum = advance_miss_gap_f32(
                csum,
                hit - rank,
                (-norm_no_hit) as f32,
                &mut best,
                &mut best_abs,
            );
        }
        let increment = 1.0 * (abs_scores[hit] * norm_hit + norm_no_hit) - norm_no_hit;
        csum += increment as f32;
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best = csum;
        }
        rank = hit + 1;
    }
    if rank < abs_scores.len() {
        advance_miss_gap_f32(
            csum,
            abs_scores.len() - rank,
            (-norm_no_hit) as f32,
            &mut best,
            &mut best_abs,
        );
    }
    best as f64
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn advance_miss_gap_f64(
    csum: &mut f64,
    first_rank: usize,
    count: usize,
    increment: f64,
    best_idx: &mut usize,
    best_value: &mut f64,
    best_abs: &mut f64,
    rmax: &mut usize,
    max_value: &mut f64,
    rmin: &mut usize,
    min_value: &mut f64,
) {
    if count == 0 {
        return;
    }

    *csum += increment;
    let first_value = *csum;
    if first_value >= *max_value {
        *max_value = first_value;
        *rmax = first_rank;
    }
    let first_abs = first_value.abs();
    if first_abs > *best_abs {
        *best_abs = first_abs;
        *best_idx = first_rank;
        *best_value = first_value;
    }

    repeat_add_assign_f64(csum, increment, count - 1);

    let last_value = *csum;
    let last_rank = first_rank + count - 1;
    if last_value <= *min_value {
        *min_value = last_value;
        *rmin = last_rank;
    }
    if count > 1 {
        let last_abs = last_value.abs();
        if last_abs > *best_abs {
            *best_abs = last_abs;
            *best_idx = last_rank;
            *best_value = last_value;
        }
    }
}

#[inline(always)]
fn advance_miss_gap_f32(
    mut csum: f32,
    count: usize,
    increment: f32,
    best: &mut f32,
    best_abs: &mut f32,
) -> f32 {
    if count == 0 {
        return csum;
    }

    csum += increment;
    let first = csum;
    let first_abs = first.abs();
    if first_abs > *best_abs {
        *best_abs = first_abs;
        *best = first;
    }

    csum = repeat_add_f32(csum, increment, count - 1);

    if count > 1 {
        let last_abs = csum.abs();
        if last_abs > *best_abs {
            *best_abs = last_abs;
            *best = csum;
        }
    }
    csum
}

#[inline(always)]
fn repeat_add_assign_f64(value: &mut f64, increment: f64, mut count: usize) {
    while count >= 16 {
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        *value += increment;
        count -= 16;
    }
    for _ in 0..count {
        *value += increment;
    }
}

#[inline(always)]
fn repeat_add_f32(mut value: f32, increment: f32, mut count: usize) -> f32 {
    while count >= 16 {
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        value += increment;
        count -= 16;
    }
    for _ in 0..count {
        value += increment;
    }
    value
}

#[cfg(test)]
fn enrichment_score_for_hits_marker_reference(
    abs_scores: &[f64],
    hits: &[usize],
    marker: &mut HitMarker,
) -> f64 {
    marker.mark_hits(hits);
    let number_hits = hits.len();
    let number_miss = abs_scores.len().saturating_sub(number_hits);
    let sum_hit_scores = hits.iter().map(|&idx| abs_scores[idx]).sum::<f64>();
    if sum_hit_scores == 0.0 || number_miss == 0 {
        return 0.0;
    }
    let norm_hit = 1.0 / sum_hit_scores;
    let norm_no_hit = 1.0 / number_miss as f64;
    let mut csum = 0.0_f32;
    let mut best = 0.0_f32;
    let mut best_abs = f32::NEG_INFINITY;
    for (i, &abs_score) in abs_scores.iter().enumerate() {
        let hit_indicator = if marker.contains(i) { 1.0 } else { 0.0 };
        let increment = hit_indicator * (abs_score * norm_hit + norm_no_hit) - norm_no_hit;
        csum += increment as f32;
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best = csum;
        }
    }
    best as f64
}

#[allow(clippy::too_many_arguments)]
fn leading_edge_blitz(
    signature: &BlitzSignature,
    hits: &PythonIntSet,
    rmax: usize,
    max_value: f64,
    rmin: usize,
    min_value: f64,
    _peak_idx: usize,
    scratch: &mut BlitzScoreScratch,
) -> Vec<String> {
    let running_len = signature.abs_scores.len();
    if running_len == 0 {
        return Vec::new();
    }
    if max_value > min_value.abs() {
        if rmax < hits.len() {
            scratch
                .leading_set
                .insert_filtered(0..rmax, |idx| hits.contains(idx));
        } else {
            scratch
                .leading_set
                .insert_filtered(hits.iter_values(), |idx| idx < rmax);
        }
    } else {
        let range_len = running_len.saturating_sub(rmin);
        if range_len < hits.len() {
            scratch
                .leading_set
                .insert_filtered(rmin..running_len, |idx| hits.contains(idx));
        } else {
            scratch
                .leading_set
                .insert_filtered(hits.iter_values(), |idx| idx >= rmin && idx < running_len);
        }
    }
    let mut leading_edge = Vec::with_capacity(scratch.leading_set.len());
    leading_edge.extend(
        scratch
            .leading_set
            .iter_values()
            .map(|idx| signature.genes[idx].clone()),
    );
    leading_edge
}

fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if !x.is_finite() || !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return f64::NAN;
    }
    crate::blitz_gamma::scipy_gammainc(alpha, x / beta)
}

#[allow(clippy::manual_range_contains)]
#[cfg(test)]
fn gamma_cdf_blitz(x: f64, alpha: f64, beta: f64, deep_accuracy: usize) -> Result<f64> {
    let prob = gamma_cdf(x, alpha, beta);
    if prob > 0.999_999_999 || prob < 0.000_000_000_01 {
        return Ok(crate::blitz_mpmath::gammacdf(x, alpha, beta, deep_accuracy)?.cdf);
    }
    Ok(prob)
}

#[allow(clippy::manual_range_contains)]
fn gamma_tail_probability_with_fallback_flag(
    branch: crate::blitz_mpmath::TailBranch,
    x: f64,
    alpha: f64,
    beta: f64,
    pos_ratio: f64,
    deep_accuracy: usize,
) -> Result<(crate::blitz_mpmath::TailProbability, bool)> {
    let prob = gamma_cdf(x, alpha, beta);
    if prob > 0.999_999_999 || prob < 0.000_000_000_01 {
        return crate::blitz_mpmath::tail_probability(
            branch,
            x,
            alpha,
            beta,
            pos_ratio,
            deep_accuracy,
        )
        .map(|tail| (tail, true));
    }
    let (prob_two_tailed, p_value) = match branch {
        crate::blitz_mpmath::TailBranch::Positive => {
            let combined = (prob * pos_ratio + 1.0 - pos_ratio).min(1.0);
            let prob_two_tailed = (1.0 - combined).min(0.5);
            (prob_two_tailed, (2.0 * prob_two_tailed).min(1.0))
        }
        crate::blitz_mpmath::TailBranch::Negative => {
            let combined = (prob - (prob * pos_ratio) + pos_ratio).min(1.0);
            let mut prob_two_tailed = (1.0 - combined).min(0.5);
            if prob_two_tailed == 0.5 {
                prob_two_tailed -= prob;
            }
            (prob_two_tailed, (2.0 * prob_two_tailed).min(1.0))
        }
    };
    Ok((
        crate::blitz_mpmath::TailProbability {
            gamma_prob: prob,
            survival_prob: 1.0 - prob,
            prob_two_tailed,
            p_value,
        },
        false,
    ))
}

fn normal_isf(p: f64) -> f64 {
    -scipy_ndtri(p)
}

#[allow(clippy::excessive_precision)]
fn scipy_ndtri(y0: f64) -> f64 {
    const EXP_NEG_TWO: f64 = 0.13533528323661269189;
    const P0: [f64; 5] = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];
    const Q0: [f64; 8] = [
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];
    const P1: [f64; 9] = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];
    const Q1: [f64; 8] = [
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];
    const P2: [f64; 9] = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];
    const Q2: [f64; 8] = [
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];

    if y0 == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y0 == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&y0) {
        return f64::NAN;
    }

    let mut code = 1;
    let mut y = y0;
    if y > 1.0 - EXP_NEG_TWO {
        y = 1.0 - y;
        code = 0;
    }

    if y > EXP_NEG_TWO {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * scipy_polevl(y2, &P0) / scipy_p1evl(y2, &Q0));
        return x * scipy_cephes_sqrtpi();
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * scipy_polevl(z, &P1) / scipy_p1evl(z, &Q1)
    } else {
        z * scipy_polevl(z, &P2) / scipy_p1evl(z, &Q2)
    };
    let x = x0 - x1;
    if code != 0 { -x } else { x }
}

#[cfg(test)]
mod tests;
