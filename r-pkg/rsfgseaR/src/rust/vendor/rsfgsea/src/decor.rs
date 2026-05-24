use crate::algo::{IntoSeed, resolve_rng_seed};
use crate::algo_support::{
    apply_bh_adjustment, build_gene_index, compute_nes, leading_edge, mode_fraction_count,
    selected_tail_count, simple_log2err, warn_prepare_stats,
};
use crate::core::{
    DecorCacheMode, DecorCorrelation, DecorOptions, DecorRedundancy, DecorWeightFormula,
    EnrichmentResult, Pathway, RankedList, ScoreType,
};
use crate::rng_compat::{RLecuyerCmrgSeedCompat, RSampleKind};
use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const CACHE_FORMAT: &str = "rsfgsea-decor-cache";
const CACHE_VERSION: &str = "1";
const GENE_ID_MODE: &str = "verbatim";
const DECOR_BATCH_MIN_GROUP_PATHWAYS: usize = 16;

#[derive(Debug, Clone)]
pub struct DecorFormulaContext {
    pub weight_formula: DecorWeightFormula,
    pub alpha: f64,
    pub gamma: f64,
    pub threshold_tau: f64,
    pub penalty_floor: f64,
    pub scale_epsilon: f64,
    pub global_median_redundancy: Option<f64>,
    pub global_q75_redundancy: Option<f64>,
    sorted_redundancy: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecorCacheStatus {
    Reused,
    Built,
    Rebuilt,
}

impl fmt::Display for DecorCacheStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCacheStatus::Reused => write!(f, "reused"),
            DecorCacheStatus::Built => write!(f, "built"),
            DecorCacheStatus::Rebuilt => write!(f, "rebuilt"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecorCacheMetadata {
    pub format: String,
    pub version: String,
    pub created_by: String,
    pub gmt_sha256: String,
    pub expression_sha256: String,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
    pub expression_gene_axis: String,
    pub expression_has_header: bool,
    pub gene_id_mode: String,
    pub n_pathways: usize,
    pub n_rows: usize,
}

#[derive(Debug, Clone)]
pub struct DecorPathwayScores {
    pub genes: Vec<String>,
    pub redundancy: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DecorCache {
    pub metadata: DecorCacheMetadata,
    pub pathways: BTreeMap<String, DecorPathwayScores>,
}

#[derive(Debug, Clone)]
pub struct DecorCacheExpectedMetadata {
    pub gmt_sha256: String,
    pub expression_sha256: Option<String>,
    pub correlation: DecorCorrelation,
    pub redundancy: DecorRedundancy,
    pub expression_gene_axis: String,
    pub expression_has_header: bool,
    pub gene_id_mode: String,
}

#[derive(Debug, Clone)]
pub struct DecorCacheCompatibility {
    pub reasons: Vec<String>,
}

impl DecorCacheCompatibility {
    pub fn is_compatible(&self) -> bool {
        self.reasons.is_empty()
    }
}

impl fmt::Display for DecorCorrelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCorrelation::Pearson => write!(f, "pearson"),
            DecorCorrelation::Spearman => write!(f, "spearman"),
        }
    }
}

impl fmt::Display for DecorRedundancy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorRedundancy::PositiveMean => write!(f, "positive_mean"),
            DecorRedundancy::AbsMean => write!(f, "abs_mean"),
        }
    }
}

impl fmt::Display for DecorCacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecorCacheMode::Auto => write!(f, "auto"),
            DecorCacheMode::Reuse => write!(f, "reuse"),
            DecorCacheMode::Rebuild => write!(f, "rebuild"),
        }
    }
}

fn parse_correlation(value: &str) -> Result<DecorCorrelation> {
    match value {
        "pearson" => Ok(DecorCorrelation::Pearson),
        "spearman" => Ok(DecorCorrelation::Spearman),
        other => bail!("unsupported decor correlation in cache metadata: {other}"),
    }
}

fn parse_redundancy(value: &str) -> Result<DecorRedundancy> {
    match value {
        "positive_mean" => Ok(DecorRedundancy::PositiveMean),
        "abs_mean" => Ok(DecorRedundancy::AbsMean),
        other => bail!("unsupported decor redundancy in cache metadata: {other}"),
    }
}

fn finite_cache_redundancy(cache: &DecorCache) -> Vec<f64> {
    let mut values = cache
        .pathways
        .values()
        .flat_map(|scores| scores.redundancy.iter().copied())
        .map(f64::from)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.total_cmp(b));
    values
}

fn percentile_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let frac = pos - lo as f64;
        Some(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
    }
}

// Keep option validation close to the formula engine so every wrapper gets the
// same numerical guardrails before ES and null generation start.
fn validate_decor_options(options: &DecorOptions) -> Result<()> {
    if options.alpha < 0.0 || !options.alpha.is_finite() {
        bail!("decor alpha must be a finite value >= 0");
    }
    if options.gamma < 0.0 || !options.gamma.is_finite() {
        bail!("decor gamma must be a finite value >= 0");
    }
    if !(0.0..1.0).contains(&options.threshold_tau) || !options.threshold_tau.is_finite() {
        bail!("decor threshold must be a finite value >= 0 and < 1");
    }
    if !(0.0..1.0).contains(&options.penalty_floor) || !options.penalty_floor.is_finite() {
        bail!("decor penalty floor must be a finite value >= 0 and < 1");
    }
    if options.scale_epsilon <= 0.0 || !options.scale_epsilon.is_finite() {
        bail!("decor scale epsilon must be a finite value > 0");
    }
    Ok(())
}

impl DecorFormulaContext {
    pub fn from_cache(cache: &DecorCache, options: &DecorOptions) -> Result<Self> {
        validate_decor_options(options)?;
        // Formula scaling is derived from the cache once per run, not per
        // pathway, so all pathways share one transparent penalty reference.
        let sorted_redundancy = finite_cache_redundancy(cache);
        let requires_distribution = matches!(
            options.weight_formula,
            DecorWeightFormula::ScaledRational
                | DecorWeightFormula::Q75ScaledRational
                | DecorWeightFormula::ExpScaled
                | DecorWeightFormula::QuantileRational
                | DecorWeightFormula::FloorScaledRational
        );
        if requires_distribution && sorted_redundancy.is_empty() {
            bail!(
                "selected formula {} could not compute redundancy scale because the decor cache has no finite redundancy scores",
                options.weight_formula
            );
        }
        Ok(Self {
            weight_formula: options.weight_formula,
            alpha: options.alpha,
            gamma: options.gamma,
            threshold_tau: options.threshold_tau,
            penalty_floor: options.penalty_floor,
            scale_epsilon: options.scale_epsilon,
            global_median_redundancy: percentile_sorted(&sorted_redundancy, 0.5),
            global_q75_redundancy: percentile_sorted(&sorted_redundancy, 0.75),
            sorted_redundancy,
        })
    }

    fn median_scale(&self) -> Result<f64> {
        self.global_median_redundancy.with_context(|| {
            format!(
                "selected formula {} requires finite redundancy scores in the decor cache",
                self.weight_formula
            )
        })
    }

    fn q75_scale(&self) -> Result<f64> {
        self.global_q75_redundancy.with_context(|| {
            format!(
                "selected formula {} requires finite redundancy scores in the decor cache",
                self.weight_formula
            )
        })
    }

    fn redundancy_quantile(&self, r: f64) -> Result<f64> {
        if self.sorted_redundancy.is_empty() {
            bail!(
                "selected formula quantile-rational requires finite redundancy scores in the decor cache"
            );
        }
        if self.sorted_redundancy.len() == 1 {
            return Ok(1.0);
        }
        let lower = self.sorted_redundancy.partition_point(|value| *value < r);
        let upper = self.sorted_redundancy.partition_point(|value| *value <= r);
        let avg_rank = (lower + upper.saturating_sub(1)) as f64 / 2.0;
        Ok((avg_rank / (self.sorted_redundancy.len() - 1) as f64).clamp(0.0, 1.0))
    }

    pub fn penalty(&self, r: f64) -> Result<f64> {
        if !r.is_finite() {
            bail!("decor redundancy score must be finite");
        }
        let penalty = match self.weight_formula {
            DecorWeightFormula::RawRational => 1.0 / (1.0 + self.alpha * r),
            DecorWeightFormula::ScaledRational => {
                let r_scaled = r / (self.median_scale()? + self.scale_epsilon);
                1.0 / (1.0 + self.alpha * r_scaled)
            }
            DecorWeightFormula::Q75ScaledRational => {
                let r_scaled = r / (self.q75_scale()? + self.scale_epsilon);
                1.0 / (1.0 + self.alpha * r_scaled)
            }
            DecorWeightFormula::ExpScaled => {
                let r_scaled = r / (self.median_scale()? + self.scale_epsilon);
                (-self.alpha * r_scaled).exp()
            }
            DecorWeightFormula::OddsRational => {
                let r = r.clamp(0.0, 1.0 - self.scale_epsilon);
                let odds = r / (1.0 - r + self.scale_epsilon);
                1.0 / (1.0 + self.alpha * odds)
            }
            DecorWeightFormula::ThresholdRational => {
                let r_star = (r - self.threshold_tau).max(0.0);
                1.0 / (1.0 + self.alpha * r_star)
            }
            DecorWeightFormula::QuantileRational => {
                let q = self.redundancy_quantile(r)?;
                1.0 / (1.0 + self.alpha * q)
            }
            DecorWeightFormula::FloorScaledRational => {
                let r_scaled = r / (self.median_scale()? + self.scale_epsilon);
                let base = 1.0 / (1.0 + self.alpha * r_scaled);
                self.penalty_floor + (1.0 - self.penalty_floor) * base
            }
            DecorWeightFormula::PowerRetention => (1.0 - r.clamp(0.0, 1.0)).powf(self.gamma),
        };
        if penalty.is_finite() && penalty >= 0.0 {
            Ok(penalty)
        } else {
            bail!(
                "selected formula {} produced an invalid penalty from redundancy {}",
                self.weight_formula,
                r
            )
        }
    }

    fn penalties_for(&self, redundancy: &[f64]) -> Result<Vec<f64>> {
        redundancy.iter().map(|&r| self.penalty(r)).collect()
    }
}

#[inline]
pub fn calculate_es_decor(
    stats: &[f64],
    hits: &[usize],
    penalty: &[f64],
    n_total: usize,
    score_type: ScoreType,
) -> Result<(f64, usize)> {
    if hits.len() != penalty.len() {
        bail!(
            "decor redundancy length mismatch: got {} redundancy values for {} hits",
            penalty.len(),
            hits.len()
        );
    }
    if hits.is_empty() {
        return Ok((0.0, 0));
    }

    let m = hits.len();
    if m == n_total {
        return Ok((0.0, hits[0]));
    }

    Ok(calculate_es_decor_prechecked(
        stats, hits, penalty, n_total, score_type,
    ))
}

#[inline]
fn calculate_es_decor_prechecked(
    stats: &[f64],
    hits: &[usize],
    penalty: &[f64],
    n_total: usize,
    score_type: ScoreType,
) -> (f64, usize) {
    debug_assert_eq!(hits.len(), penalty.len());
    if hits.is_empty() {
        return (0.0, 0);
    }

    let m = hits.len();
    if m == n_total {
        return (0.0, hits[0]);
    }

    let mut nr = 0.0_f64;
    for i in 0..m {
        nr += stats[hits[i]].abs() * penalty[i];
    }

    let mut max_p = f64::NEG_INFINITY;
    let mut min_p = f64::INFINITY;
    let mut max_i = 0usize;
    let mut min_i = 0usize;
    let mut csum = 0.0;
    let denom = (n_total - m) as f64;
    let inv_m = 1.0 / m as f64;
    let nr_is_zero = nr == 0.0;

    for i in 0..m {
        let adj_i = stats[hits[i]].abs() * penalty[i];
        csum += adj_i;
        let r_cum = if nr_is_zero {
            (i + 1) as f64 * inv_m
        } else {
            csum / nr
        };
        let miss = (hits[i] - i) as f64 / denom;
        let top = r_cum - miss;
        let bottom = if nr_is_zero {
            top - inv_m
        } else {
            top - adj_i / nr
        };
        if top > max_p {
            max_p = top;
            max_i = i;
        }
        if bottom < min_p {
            min_p = bottom;
            min_i = i;
        }
    }

    match score_type {
        ScoreType::Std => {
            if max_p == -min_p {
                (0.0, hits[0])
            } else if max_p > -min_p {
                (max_p, hits[max_i])
            } else {
                (min_p, hits[min_i])
            }
        }
        ScoreType::Pos => (max_p, hits[max_i]),
        ScoreType::Neg => (min_p, hits[min_i]),
    }
}

#[inline]
fn decor_pathway_seed(seed: u64, pathway_idx: usize) -> u32 {
    let mut x = seed ^ ((pathway_idx as u64).wrapping_add(0x9e37_79b9_7f4a_7c15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    let mixed = (x as u32) ^ ((x >> 32) as u32);
    if mixed == 0 { 1 } else { mixed }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DecorNullCounts {
    n_le_es: usize,
    n_ge_es: usize,
    n_le_zero: usize,
    n_ge_zero: usize,
    le_zero_sum: f64,
    ge_zero_sum: f64,
}

#[derive(Default)]
struct DecorBatchScratch {
    nr: Vec<f64>,
    csum: Vec<f64>,
    max_p: Vec<f64>,
    min_p: Vec<f64>,
}

struct DecorWorking {
    pathway_name: String,
    size: usize,
    hits: Vec<usize>,
    penalty: Vec<f64>,
    es: f64,
    peak_idx: usize,
    n_le_es: usize,
    n_ge_es: usize,
    n_le_zero: usize,
    n_ge_zero: usize,
    le_zero_sum: f64,
    ge_zero_sum: f64,
    nes: Option<f64>,
    p_value: f64,
    padj: Option<f64>,
    log2err: Option<f64>,
}

struct DecorRuntimeSizeGroup {
    size: usize,
    work_indices: Vec<usize>,
    observed_es: Vec<f64>,
    penalties_rank_major: Vec<f64>,
    use_batched: bool,
}

impl DecorRuntimeSizeGroup {
    fn from_indices(size: usize, work_indices: Vec<usize>, work: &[DecorWorking]) -> Self {
        let use_batched = work_indices.len() >= DECOR_BATCH_MIN_GROUP_PATHWAYS;
        let observed_es = if use_batched {
            work_indices
                .iter()
                .map(|&idx| work[idx].es)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut penalties_rank_major = if use_batched {
            Vec::with_capacity(size * work_indices.len())
        } else {
            Vec::new()
        };
        if use_batched {
            for rank_idx in 0..size {
                for &work_idx in &work_indices {
                    penalties_rank_major.push(work[work_idx].penalty[rank_idx]);
                }
            }
        }
        Self {
            size,
            work_indices,
            observed_es,
            penalties_rank_major,
            use_batched,
        }
    }
}

impl DecorBatchScratch {
    fn reset(&mut self, n_pathways: usize) {
        self.nr.clear();
        self.nr.resize(n_pathways, 0.0);
        self.csum.clear();
        self.csum.resize(n_pathways, 0.0);
        self.max_p.clear();
        self.max_p.resize(n_pathways, f64::NEG_INFINITY);
        self.min_p.clear();
        self.min_p.resize(n_pathways, f64::INFINITY);
    }
}

#[allow(clippy::too_many_arguments)]
fn update_decor_batched_counts(
    stats_abs: &[f64],
    selected: &[usize],
    penalties_rank_major: &[f64],
    observed_es: &[f64],
    n_total: usize,
    score_type: ScoreType,
    counts: &mut [DecorNullCounts],
    scratch: &mut DecorBatchScratch,
) {
    let m = selected.len();
    let n_pathways = observed_es.len();
    debug_assert_eq!(counts.len(), n_pathways);
    debug_assert_eq!(penalties_rank_major.len(), m * n_pathways);
    if m == 0 || n_pathways == 0 {
        return;
    }

    scratch.reset(n_pathways);
    for (rank_idx, &hit_idx) in selected.iter().enumerate() {
        let base = stats_abs[hit_idx];
        let row = &penalties_rank_major[rank_idx * n_pathways..(rank_idx + 1) * n_pathways];
        for (nr, &penalty) in scratch.nr.iter_mut().zip(row) {
            *nr += base * penalty;
        }
    }

    let denom = (n_total - m) as f64;
    let inv_m = 1.0 / m as f64;
    for (rank_idx, &hit_idx) in selected.iter().enumerate() {
        let base = stats_abs[hit_idx];
        let miss = (hit_idx - rank_idx) as f64 / denom;
        let row = &penalties_rank_major[rank_idx * n_pathways..(rank_idx + 1) * n_pathways];

        for (pathway_idx, &penalty) in row.iter().enumerate() {
            let adj = base * penalty;
            scratch.csum[pathway_idx] += adj;
            let nr = scratch.nr[pathway_idx];
            let top = if nr == 0.0 {
                (rank_idx + 1) as f64 * inv_m - miss
            } else {
                scratch.csum[pathway_idx] / nr - miss
            };
            let bottom = if nr == 0.0 {
                top - inv_m
            } else {
                top - adj / nr
            };
            if top > scratch.max_p[pathway_idx] {
                scratch.max_p[pathway_idx] = top;
            }
            if bottom < scratch.min_p[pathway_idx] {
                scratch.min_p[pathway_idx] = bottom;
            }
        }
    }

    match score_type {
        ScoreType::Std => {
            for pathway_idx in 0..n_pathways {
                let rand_es = if scratch.max_p[pathway_idx] == -scratch.min_p[pathway_idx] {
                    0.0
                } else if scratch.max_p[pathway_idx] > -scratch.min_p[pathway_idx] {
                    scratch.max_p[pathway_idx]
                } else {
                    scratch.min_p[pathway_idx]
                };
                update_decor_null_count(
                    &mut counts[pathway_idx],
                    rand_es,
                    observed_es[pathway_idx],
                );
            }
        }
        ScoreType::Pos => {
            for pathway_idx in 0..n_pathways {
                update_decor_null_count(
                    &mut counts[pathway_idx],
                    scratch.max_p[pathway_idx],
                    observed_es[pathway_idx],
                );
            }
        }
        ScoreType::Neg => {
            for pathway_idx in 0..n_pathways {
                update_decor_null_count(
                    &mut counts[pathway_idx],
                    scratch.min_p[pathway_idx],
                    observed_es[pathway_idx],
                );
            }
        }
    }
}

#[inline]
fn update_decor_null_count(count: &mut DecorNullCounts, rand_es: f64, observed_es: f64) {
    if rand_es <= observed_es {
        count.n_le_es += 1;
    }
    if rand_es >= observed_es {
        count.n_ge_es += 1;
    }
    if rand_es <= 0.0 {
        count.n_le_zero += 1;
        count.le_zero_sum += rand_es;
    }
    if rand_es >= 0.0 {
        count.n_ge_zero += 1;
        count.ge_zero_sum += rand_es;
    }
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("Failed to read '{}'", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(hex)
}

pub fn validate_decor_cache(
    cache_metadata: &DecorCacheMetadata,
    expected: &DecorCacheExpectedMetadata,
) -> DecorCacheCompatibility {
    let mut reasons = Vec::new();
    if cache_metadata.format != CACHE_FORMAT {
        reasons.push(format!(
            "cache format differs: cache={}",
            cache_metadata.format
        ));
    }
    if cache_metadata.version != CACHE_VERSION {
        reasons.push(format!(
            "cache format version is unsupported: cache={}",
            cache_metadata.version
        ));
    }
    if cache_metadata.gmt_sha256 != expected.gmt_sha256 {
        reasons.push("GMT SHA256 differs".to_string());
    }
    if let Some(expression_sha256) = &expected.expression_sha256
        && cache_metadata.expression_sha256 != *expression_sha256
    {
        reasons.push("expression SHA256 differs".to_string());
    }
    if cache_metadata.correlation != expected.correlation {
        reasons.push(format!(
            "correlation method differs: cache={}, requested={}",
            cache_metadata.correlation, expected.correlation
        ));
    }
    if cache_metadata.redundancy != expected.redundancy {
        reasons.push(format!(
            "redundancy mode differs: cache={}, requested={}",
            cache_metadata.redundancy, expected.redundancy
        ));
    }
    if cache_metadata.expression_gene_axis != expected.expression_gene_axis {
        reasons.push(format!(
            "expression gene axis differs: cache={}, requested={}",
            cache_metadata.expression_gene_axis, expected.expression_gene_axis
        ));
    }
    if cache_metadata.expression_has_header != expected.expression_has_header {
        reasons.push(format!(
            "expression header setting differs: cache={}, requested={}",
            cache_metadata.expression_has_header, expected.expression_has_header
        ));
    }
    if cache_metadata.gene_id_mode != expected.gene_id_mode {
        reasons.push(format!(
            "gene ID mode differs: cache={}, requested={}",
            cache_metadata.gene_id_mode, expected.gene_id_mode
        ));
    }
    DecorCacheCompatibility { reasons }
}

pub fn ensure_decor_cache_for_paths(
    pathways: &[Pathway],
    gmt_path: &Path,
    options: &DecorOptions,
    expression_has_header: bool,
) -> Result<(DecorCache, DecorCacheStatus)> {
    validate_decor_options(options)?;
    if options.correlation == DecorCorrelation::Spearman {
        bail!("spearman decor correlation is not implemented yet");
    }

    let cache_path = options
        .cache_path
        .as_deref()
        .context("method decor requires --decor-cache")?;
    let gmt_sha256 = file_sha256(gmt_path)?;
    let expression_sha256 = options
        .expression_path
        .as_deref()
        .map(file_sha256)
        .transpose()?;
    let expected = DecorCacheExpectedMetadata {
        gmt_sha256,
        expression_sha256: expression_sha256.clone(),
        correlation: options.correlation,
        redundancy: options.redundancy,
        expression_gene_axis: "rows".to_string(),
        expression_has_header,
        gene_id_mode: GENE_ID_MODE.to_string(),
    };

    // Cache identity is content-based: GMT, expression matrix, correlation, and
    // redundancy mode must match, while formula presets can change freely.
    let cache_exists = cache_path.exists();
    if cache_exists && options.cache_mode != DecorCacheMode::Rebuild {
        match read_decor_cache(cache_path) {
            Ok(cache) => {
                let compatibility = validate_decor_cache(&cache.metadata, &expected);
                if compatibility.is_compatible() {
                    return Ok((cache, DecorCacheStatus::Reused));
                }
                if options.cache_mode == DecorCacheMode::Reuse {
                    bail!("{}", incompatible_message(&compatibility, true));
                }
                if options.expression_path.is_none() {
                    bail!("{}", incompatible_message(&compatibility, false));
                }
            }
            Err(err) => {
                if options.cache_mode == DecorCacheMode::Reuse {
                    bail!("Decor cache could not be read: {err}");
                }
                if options.expression_path.is_none() {
                    bail!(
                        "Decor cache could not be read and --decor-expression was not provided: {err}"
                    );
                }
            }
        }
    } else if !cache_exists && options.cache_mode == DecorCacheMode::Reuse {
        bail!("decor cache does not exist: {}", cache_path.display());
    }

    let expression_path = options.expression_path.as_deref().context(
        "decor cache does not exist or is incompatible and --decor-expression was not provided",
    )?;
    let expression_sha256 = expression_sha256.context("internal error: missing expression hash")?;
    let cache = build_decor_cache_from_expression(
        pathways,
        expression_path,
        DecorCacheExpectedMetadata {
            expression_sha256: Some(expression_sha256),
            ..expected
        },
    )?;
    write_decor_cache_atomic(cache_path, &cache)?;
    let status = if cache_exists {
        DecorCacheStatus::Rebuilt
    } else {
        DecorCacheStatus::Built
    };
    Ok((cache, status))
}

fn incompatible_message(compatibility: &DecorCacheCompatibility, suggest_rebuild: bool) -> String {
    let mut msg = String::from("Decor cache is incompatible:");
    for reason in &compatibility.reasons {
        msg.push_str("\n  - ");
        msg.push_str(reason);
    }
    if suggest_rebuild {
        msg.push_str("\nUse --decor-cache-mode rebuild with --decor-expression to rebuild it.");
    } else {
        msg.push_str(
            "\nProvide --decor-expression or use --decor-cache-mode rebuild to rebuild it.",
        );
    }
    msg
}

#[derive(Debug)]
struct ExpressionMatrix {
    genes: Vec<String>,
    values: Vec<Vec<f64>>,
}

fn read_expression_matrix(path: &Path, has_header: bool) -> Result<ExpressionMatrix> {
    let file = File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut genes = Vec::new();
    let mut values = Vec::new();
    let mut seen = HashSet::new();
    let mut expected_cols = None;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if has_header && line_idx == 0 {
            let cols = line.split('\t').count();
            if cols < 3 {
                bail!("expression matrix header must contain gene and at least two sample columns");
            }
            expected_cols = Some(cols - 1);
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            bail!(
                "Malformed expression matrix line {}: expected gene and at least two sample columns",
                line_idx + 1
            );
        }
        if let Some(cols) = expected_cols {
            if fields.len() - 1 != cols {
                bail!(
                    "Malformed expression matrix line {}: expected {} sample columns, found {}",
                    line_idx + 1,
                    cols,
                    fields.len() - 1
                );
            }
        } else {
            expected_cols = Some(fields.len() - 1);
        }

        let gene = fields[0].to_string();
        if gene.is_empty() {
            bail!(
                "expression matrix line {} has an empty gene identifier",
                line_idx + 1
            );
        }
        if !seen.insert(gene.clone()) {
            bail!("expression matrix has duplicate gene identifier: {gene}");
        }
        let row: Vec<f64> = fields[1..]
            .iter()
            .map(|value| {
                value.parse::<f64>().with_context(|| {
                    format!(
                        "Failed to parse expression value '{}' on line {}",
                        value,
                        line_idx + 1
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if row.iter().any(|value| !value.is_finite()) {
            bail!(
                "expression matrix line {} contains a non-finite value",
                line_idx + 1
            );
        }
        genes.push(gene);
        values.push(row);
    }

    if values.is_empty() {
        bail!("expression matrix contains no gene rows");
    }
    if expected_cols.unwrap_or(0) < 2 {
        bail!("expression matrix must contain at least two sample columns");
    }
    Ok(ExpressionMatrix { genes, values })
}

fn standardized_rows(matrix: ExpressionMatrix) -> HashMap<String, Vec<f64>> {
    matrix
        .genes
        .into_iter()
        .zip(matrix.values)
        .map(|(gene, values)| {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let centered: Vec<f64> = values.iter().map(|value| value - mean).collect();
            let ss = centered.iter().map(|value| value * value).sum::<f64>();
            if ss <= 0.0 {
                (gene, vec![0.0; centered.len()])
            } else {
                let scale = ss.sqrt();
                (
                    gene,
                    centered.into_iter().map(|value| value / scale).collect(),
                )
            }
        })
        .collect()
}

pub fn build_decor_cache_from_expression(
    pathways: &[Pathway],
    expression_path: &Path,
    expected: DecorCacheExpectedMetadata,
) -> Result<DecorCache> {
    if expected.correlation == DecorCorrelation::Spearman {
        bail!("spearman decor correlation is not implemented yet");
    }
    let expression = read_expression_matrix(expression_path, expected.expression_has_header)?;
    let standardized = standardized_rows(expression);

    // Store per-gene redundancy in pathway order so later runs only need ranks
    // and the cache; expression is not re-read when the cache is compatible.
    let mut rows: Vec<(String, DecorPathwayScores)> = pathways
        .par_iter()
        .map(|pathway| {
            let mut genes = Vec::new();
            let mut vectors = Vec::new();
            let mut seen = HashSet::new();
            for gene in &pathway.genes {
                if !seen.insert(gene) {
                    continue;
                }
                if let Some(vector) = standardized.get(gene) {
                    genes.push(gene.clone());
                    vectors.push(vector.as_slice());
                }
            }
            let redundancy = compute_redundancy(&vectors, expected.redundancy);
            (
                pathway.name.clone(),
                DecorPathwayScores { genes, redundancy },
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let n_rows = rows.iter().map(|(_, row)| row.genes.len()).sum();
    let metadata = DecorCacheMetadata {
        format: CACHE_FORMAT.to_string(),
        version: CACHE_VERSION.to_string(),
        created_by: "rsfgsea".to_string(),
        gmt_sha256: expected.gmt_sha256,
        expression_sha256: expected
            .expression_sha256
            .unwrap_or_else(|| "unknown".to_string()),
        correlation: expected.correlation,
        redundancy: expected.redundancy,
        expression_gene_axis: expected.expression_gene_axis,
        expression_has_header: expected.expression_has_header,
        gene_id_mode: expected.gene_id_mode,
        n_pathways: rows.len(),
        n_rows,
    };
    Ok(DecorCache {
        metadata,
        pathways: rows.into_iter().collect(),
    })
}

fn compute_redundancy(vectors: &[&[f64]], mode: DecorRedundancy) -> Vec<f32> {
    let m = vectors.len();
    if m < 2 {
        return vec![0.0; m];
    }
    let mut sums = vec![0.0_f64; m];
    for i in 0..m {
        for j in i + 1..m {
            let corr = vectors[i]
                .iter()
                .zip(vectors[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            let value = match mode {
                DecorRedundancy::PositiveMean => corr.max(0.0),
                DecorRedundancy::AbsMean => corr.abs(),
            };
            sums[i] += value;
            sums[j] += value;
        }
    }
    let denom = (m - 1) as f64;
    sums.into_iter()
        .map(|sum| (sum / denom).clamp(0.0, 1.0) as f32)
        .collect()
}

pub fn read_decor_cache(path: &Path) -> Result<DecorCache> {
    let file = File::open(path).with_context(|| format!("Failed to open '{}'", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut metadata_map = BTreeMap::new();
    let mut pathways: BTreeMap<String, DecorPathwayScores> = BTreeMap::new();
    let mut saw_header = false;
    let mut row_count = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let Some((key, value)) = comment.trim().split_once('=') else {
                continue;
            };
            metadata_map.insert(key.trim().to_string(), value.trim().to_string());
            continue;
        }
        if !saw_header {
            if line != "pathway\tgene\tredundancy" {
                bail!("decor cache is missing required pathway/gene/redundancy header");
            }
            saw_header = true;
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            bail!(
                "Malformed decor cache line {}: expected pathway, gene, redundancy",
                line_idx + 1
            );
        }
        let redundancy: f32 = fields[2].parse().with_context(|| {
            format!(
                "Failed to parse redundancy '{}' on cache line {}",
                fields[2],
                line_idx + 1
            )
        })?;
        if !redundancy.is_finite() {
            bail!(
                "decor cache line {} has non-finite redundancy",
                line_idx + 1
            );
        }
        let entry = pathways
            .entry(fields[0].to_string())
            .or_insert_with(|| DecorPathwayScores {
                genes: Vec::new(),
                redundancy: Vec::new(),
            });
        entry.genes.push(fields[1].to_string());
        entry.redundancy.push(redundancy.clamp(0.0, 1.0));
        row_count += 1;
    }

    if !saw_header {
        bail!("decor cache is missing tabular header");
    }
    let metadata = parse_cache_metadata(&metadata_map)?;
    if metadata.n_rows != row_count {
        bail!(
            "decor cache row count metadata mismatch: metadata={}, observed={}",
            metadata.n_rows,
            row_count
        );
    }
    Ok(DecorCache { metadata, pathways })
}

fn required_meta<'a>(map: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    map.get(key)
        .map(String::as_str)
        .with_context(|| format!("decor cache is missing metadata key: {key}"))
}

fn parse_cache_metadata(map: &BTreeMap<String, String>) -> Result<DecorCacheMetadata> {
    Ok(DecorCacheMetadata {
        format: required_meta(map, "rsfgsea_decor_cache_format")?.to_string(),
        version: required_meta(map, "rsfgsea_decor_cache_version")?.to_string(),
        created_by: required_meta(map, "created_by")?.to_string(),
        gmt_sha256: required_meta(map, "gmt_sha256")?.to_string(),
        expression_sha256: required_meta(map, "expression_sha256")?.to_string(),
        correlation: parse_correlation(required_meta(map, "correlation")?)?,
        redundancy: parse_redundancy(required_meta(map, "redundancy")?)?,
        expression_gene_axis: required_meta(map, "expression_gene_axis")?.to_string(),
        expression_has_header: required_meta(map, "expression_has_header")?.parse()?,
        gene_id_mode: required_meta(map, "gene_id_mode")?.to_string(),
        n_pathways: required_meta(map, "n_pathways")?.parse()?,
        n_rows: required_meta(map, "n_rows")?.parse()?,
    })
}

pub fn write_decor_cache_atomic(path: &Path, cache: &DecorCache) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create cache directory '{}'", parent.display()))?;
    }
    let pid = std::process::id();
    let tmp_path = tmp_cache_path(path, pid);
    let result = (|| -> Result<()> {
        let file = File::create(&tmp_path)
            .with_context(|| format!("Failed to create '{}'", tmp_path.display()))?;
        if path.extension().is_some_and(|ext| ext == "gz") {
            let mut writer = GzEncoder::new(file, Compression::default());
            write_cache_contents(&mut writer, cache)?;
            writer.finish()?.sync_all()?;
        } else {
            let mut writer = BufWriter::new(file);
            write_cache_contents(&mut writer, cache)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "Failed to rename temporary cache '{}' to '{}'",
                tmp_path.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

fn tmp_cache_path(path: &Path, pid: u32) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("decor-cache");
    path.with_file_name(format!("{file_name}.tmp.{pid}"))
}

fn write_cache_contents<W: Write>(writer: &mut W, cache: &DecorCache) -> Result<()> {
    let md = &cache.metadata;
    writeln!(writer, "# rsfgsea_decor_cache_format={}", md.format)?;
    writeln!(writer, "# rsfgsea_decor_cache_version={}", md.version)?;
    writeln!(writer, "# created_by={}", md.created_by)?;
    writeln!(writer, "# gmt_sha256={}", md.gmt_sha256)?;
    writeln!(writer, "# expression_sha256={}", md.expression_sha256)?;
    writeln!(writer, "# correlation={}", md.correlation)?;
    writeln!(writer, "# redundancy={}", md.redundancy)?;
    writeln!(writer, "# expression_gene_axis={}", md.expression_gene_axis)?;
    writeln!(
        writer,
        "# expression_has_header={}",
        md.expression_has_header
    )?;
    writeln!(writer, "# gene_id_mode={}", md.gene_id_mode)?;
    writeln!(writer, "# n_pathways={}", md.n_pathways)?;
    writeln!(writer, "# n_rows={}", md.n_rows)?;
    writeln!(writer, "pathway\tgene\tredundancy")?;
    for (pathway, scores) in &cache.pathways {
        for (gene, redundancy) in scores.genes.iter().zip(scores.redundancy.iter()) {
            writeln!(writer, "{pathway}\t{gene}\t{redundancy:.8}")?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_decor_simple_with_sample_size<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    cache: &DecorCache,
    alpha: f64,
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    _sample_size: usize,
) -> Result<Vec<EnrichmentResult>> {
    let options = DecorOptions {
        alpha,
        ..DecorOptions::default()
    };
    fgsea_decor_simple_with_options(
        ranks,
        pathways,
        cache,
        &options,
        n_perm,
        seed,
        min_size,
        max_size,
        eps,
        score_type,
        gsea_param,
        _sample_size,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fgsea_decor_simple_with_options<S: IntoSeed>(
    ranks: &RankedList,
    pathways: &[Pathway],
    cache: &DecorCache,
    options: &DecorOptions,
    n_perm: usize,
    seed: S,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
    _sample_size: usize,
) -> Result<Vec<EnrichmentResult>> {
    let seed = resolve_rng_seed(seed.into_seed());
    run_decor_simple_internal(
        ranks, pathways, cache, options, n_perm, seed, min_size, max_size, eps, score_type,
        gsea_param,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_decor_simple_internal(
    ranks: &RankedList,
    pathways: &[Pathway],
    cache: &DecorCache,
    options: &DecorOptions,
    n_perm: usize,
    seed: u64,
    min_size: usize,
    max_size: usize,
    eps: f64,
    score_type: ScoreType,
    gsea_param: f64,
) -> Result<Vec<EnrichmentResult>> {
    let formula_context = DecorFormulaContext::from_cache(cache, options)?;
    let gene_to_idx = build_gene_index(ranks);
    let n_total = ranks.len();
    warn_prepare_stats(ranks, score_type);
    let min_size = min_size.max(1);
    let max_size = max_size.min(n_total.saturating_sub(1));
    let eps = eps.clamp(0.0, 1.0);
    let (_abs_weights, scaled_scores, _ns_total) = ranks.prepare(gsea_param);
    let simple_stats: Vec<f64> = scaled_scores.iter().map(|&v| v as f64).collect();

    // Observed pathways keep their pathway-specific decor penalties; the
    // permutation calibration below preserves pathway size while drawing rank
    // positions from the same score profile.
    let mut work: Vec<DecorWorking> = pathways
        .par_iter()
        .map(|pw| {
            let (hits, redundancy) = extract_decor_hits(pw, &gene_to_idx, cache);
            if hits.len() < min_size || hits.len() > max_size {
                return Ok(None);
            }
            let penalty = formula_context.penalties_for(&redundancy)?;
            let (es, peak_idx) =
                calculate_es_decor_prechecked(&simple_stats, &hits, &penalty, n_total, score_type);
            Ok(Some(DecorWorking {
                pathway_name: pw.name.clone(),
                size: hits.len(),
                hits,
                penalty,
                es,
                peak_idx,
                n_le_es: 0,
                n_ge_es: 0,
                n_le_zero: 0,
                n_ge_zero: 0,
                le_zero_sum: 0.0,
                ge_zero_sum: 0.0,
                nes: None,
                p_value: f64::NAN,
                padj: None,
                log2err: None,
            }))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    if n_perm > 0 && !work.is_empty() {
        let simple_stats_abs: Vec<f64> = simple_stats.iter().map(|v| v.abs()).collect();
        let mut size_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (idx, w) in work.iter().enumerate() {
            size_groups.entry(w.size).or_default().push(idx);
        }
        let runtime_groups = size_groups
            .into_iter()
            .map(|(size, indices)| DecorRuntimeSizeGroup::from_indices(size, indices, &work))
            .collect::<Vec<_>>();

        // Size groups reuse sampled hit-position profiles across pathways with
        // identical cardinality, while batched groups avoid repeated ES scans.
        let grouped_counts = runtime_groups
            .par_iter()
            .enumerate()
            .map(|(group_idx, group)| {
                let n_pathways = group.work_indices.len();
                let mut counts = vec![DecorNullCounts::default(); n_pathways];
                let mut rng =
                    RLecuyerCmrgSeedCompat::from_r_set_seed(decor_pathway_seed(seed, group.size));
                let mut scratch = DecorBatchScratch::default();
                for _ in 0..n_perm {
                    let mut selected = rng.sample_int_no_replace_with_kind(
                        n_total,
                        group.size,
                        RSampleKind::Rejection,
                    );
                    for idx in &mut selected {
                        *idx -= 1;
                    }
                    selected.sort_unstable();
                    if group.use_batched {
                        update_decor_batched_counts(
                            &simple_stats_abs,
                            &selected,
                            &group.penalties_rank_major,
                            &group.observed_es,
                            n_total,
                            score_type,
                            &mut counts,
                            &mut scratch,
                        );
                    } else {
                        for (count, work_idx) in counts.iter_mut().zip(group.work_indices.iter()) {
                            let w = &work[*work_idx];
                            let (rand_es, _) = calculate_es_decor_prechecked(
                                &simple_stats,
                                &selected,
                                &w.penalty,
                                n_total,
                                score_type,
                            );
                            update_decor_null_count(count, rand_es, w.es);
                        }
                    }
                }
                (group_idx, counts)
            })
            .collect::<Vec<_>>();

        for (group_idx, counts) in grouped_counts {
            let group = &runtime_groups[group_idx];
            for (&work_idx, counts) in group.work_indices.iter().zip(counts) {
                let w = &mut work[work_idx];
                w.n_le_es = counts.n_le_es;
                w.n_ge_es = counts.n_ge_es;
                w.n_le_zero = counts.n_le_zero;
                w.n_ge_zero = counts.n_ge_zero;
                w.le_zero_sum = counts.le_zero_sum;
                w.ge_zero_sum = counts.ge_zero_sum;
            }
        }
    }

    for w in &mut work {
        let le_zero_mean = if w.n_le_zero > 0 {
            w.le_zero_sum / w.n_le_zero as f64
        } else {
            0.0
        };
        let ge_zero_mean = if w.n_ge_zero > 0 {
            w.ge_zero_sum / w.n_ge_zero as f64
        } else {
            0.0
        };
        w.nes = compute_nes(w.es, score_type, le_zero_mean, ge_zero_mean);
        if w.nes.is_some() {
            let p_le = (w.n_le_es + 1) as f64 / (w.n_le_zero + 1) as f64;
            let p_ge = (w.n_ge_es + 1) as f64 / (w.n_ge_zero + 1) as f64;
            w.p_value = p_le.min(p_ge).max(eps);
        }
        let n_more_extreme =
            selected_tail_count(score_type, w.es, w.n_le_es as u64, w.n_ge_es as u64);
        let mode_fraction =
            mode_fraction_count(score_type, w.es, w.n_le_zero as u64, w.n_ge_zero as u64);
        w.log2err = if w.p_value.is_finite() && mode_fraction > 0 {
            simple_log2err(n_more_extreme, n_perm)
        } else {
            None
        };
    }

    let mut final_results: Vec<EnrichmentResult> = work
        .into_iter()
        .map(|w| EnrichmentResult {
            pathway_name: w.pathway_name,
            size: w.size,
            es: w.es,
            nes: w.nes,
            p_value: w.p_value,
            padj: w.padj,
            log2err: w.log2err,
            leading_edge: leading_edge(&w.hits, w.peak_idx, w.es, score_type, ranks),
        })
        .collect();
    apply_bh_adjustment(&mut final_results);
    final_results.sort_by(|a, b| a.pathway_name.cmp(&b.pathway_name));
    Ok(final_results)
}

fn extract_decor_hits(
    pathway: &Pathway,
    gene_to_idx: &HashMap<String, usize>,
    cache: &DecorCache,
) -> (Vec<usize>, Vec<f64>) {
    let Some(scores) = cache.pathways.get(&pathway.name) else {
        return (Vec::new(), Vec::new());
    };
    let redundancy_by_gene: HashMap<&str, f64> = scores
        .genes
        .iter()
        .zip(scores.redundancy.iter())
        .map(|(gene, redundancy)| (gene.as_str(), *redundancy as f64))
        .collect();
    let mut hit_pairs: Vec<(usize, f64)> = pathway
        .genes
        .iter()
        .filter_map(|gene| {
            let idx = gene_to_idx.get(gene)?;
            let redundancy = redundancy_by_gene.get(gene.as_str())?;
            Some((*idx, *redundancy))
        })
        .collect();
    hit_pairs.sort_by_key(|(idx, _)| *idx);
    hit_pairs.dedup_by_key(|(idx, _)| *idx);
    hit_pairs.into_iter().unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::calculate_es_fgsea;
    use tempfile::tempdir;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-8,
            "actual={actual}, expected={expected}"
        );
    }

    fn formula_cache(values: &[f32]) -> DecorCache {
        DecorCache {
            metadata: DecorCacheMetadata {
                format: CACHE_FORMAT.to_string(),
                version: CACHE_VERSION.to_string(),
                created_by: "rsfgsea".to_string(),
                gmt_sha256: "gmt".to_string(),
                expression_sha256: "expr".to_string(),
                correlation: DecorCorrelation::Pearson,
                redundancy: DecorRedundancy::PositiveMean,
                expression_gene_axis: "rows".to_string(),
                expression_has_header: true,
                gene_id_mode: GENE_ID_MODE.to_string(),
                n_pathways: 1,
                n_rows: values.len(),
            },
            pathways: BTreeMap::from([(
                "PW".to_string(),
                DecorPathwayScores {
                    genes: values
                        .iter()
                        .enumerate()
                        .map(|(idx, _)| format!("G{idx}"))
                        .collect(),
                    redundancy: values.to_vec(),
                },
            )]),
        }
    }

    fn formula_context(formula: DecorWeightFormula) -> DecorFormulaContext {
        let cache = formula_cache(&[0.0, 0.2, 0.4, 0.8]);
        let options = DecorOptions {
            alpha: 2.0,
            weight_formula: formula,
            ..DecorOptions::default()
        };
        DecorFormulaContext::from_cache(&cache, &options).unwrap()
    }

    #[test]
    fn decor_formula_raw_rational_matches_manual_values() {
        let ctx = formula_context(DecorWeightFormula::RawRational);
        assert_close(ctx.penalty(0.0).unwrap(), 1.0);
        assert_close(ctx.penalty(0.5).unwrap(), 0.5);

        let cache = formula_cache(&[0.5]);
        let options = DecorOptions {
            alpha: 0.0,
            weight_formula: DecorWeightFormula::RawRational,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert_close(ctx.penalty(0.9).unwrap(), 1.0);
    }

    #[test]
    fn decor_formula_scaled_rational_uses_global_median() {
        let ctx = formula_context(DecorWeightFormula::ScaledRational);
        assert_close(ctx.global_median_redundancy.unwrap(), 0.3);
        let expected = 1.0 / (1.0 + 2.0 * (0.6 / (0.3 + 1e-12)));
        assert_close(ctx.penalty(0.6).unwrap(), expected);
    }

    #[test]
    fn decor_formula_q75_scaled_rational_uses_global_q75() {
        let ctx = formula_context(DecorWeightFormula::Q75ScaledRational);
        assert_close(ctx.global_q75_redundancy.unwrap(), 0.5);
        let expected = 1.0 / (1.0 + 2.0 * (0.5 / (0.5 + 1e-12)));
        assert_close(ctx.penalty(0.5).unwrap(), expected);
    }

    #[test]
    fn decor_formula_exp_scaled_matches_manual_value() {
        let ctx = formula_context(DecorWeightFormula::ExpScaled);
        let expected = (-2.0_f64 * (0.3_f64 / (0.3_f64 + 1e-12_f64))).exp();
        assert_close(ctx.penalty(0.3).unwrap(), expected);
    }

    #[test]
    fn decor_formula_odds_rational_emphasizes_high_redundancy() {
        let ctx = formula_context(DecorWeightFormula::OddsRational);
        assert_close(ctx.penalty(0.0).unwrap(), 1.0);
        assert_close(
            ctx.penalty(0.5).unwrap(),
            1.0 / (1.0 + 2.0 * (0.5 / (0.5 + 1e-12))),
        );
        assert!(ctx.penalty(1.0).unwrap() < 1e-6);
    }

    #[test]
    fn decor_formula_threshold_rational_ignores_scores_below_tau() {
        let cache = formula_cache(&[0.1, 0.2, 0.5]);
        let options = DecorOptions {
            alpha: 2.0,
            weight_formula: DecorWeightFormula::ThresholdRational,
            threshold_tau: 0.25,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert_close(ctx.penalty(0.2).unwrap(), 1.0);
        assert_close(ctx.penalty(0.5).unwrap(), 1.0 / (1.0 + 2.0 * 0.25));
    }

    #[test]
    fn decor_formula_quantile_rational_uses_average_rank_for_ties() {
        let cache = formula_cache(&[0.1, 0.2, 0.2, 0.8]);
        let options = DecorOptions {
            alpha: 2.0,
            weight_formula: DecorWeightFormula::QuantileRational,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        let q = 1.5 / 3.0;
        assert_close(
            ctx.penalty(f64::from(0.2_f32)).unwrap(),
            1.0 / (1.0 + 2.0 * q),
        );
    }

    #[test]
    fn decor_formula_floor_scaled_rational_respects_floor() {
        let cache = formula_cache(&[0.0, 0.2, 0.4, 0.8]);
        let options = DecorOptions {
            alpha: 1000.0,
            weight_formula: DecorWeightFormula::FloorScaledRational,
            penalty_floor: 0.25,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert!(ctx.penalty(0.8).unwrap() >= 0.25);
        assert_close(ctx.penalty(0.0).unwrap(), 1.0);
    }

    #[test]
    fn decor_formula_power_retention_uses_gamma() {
        let cache = formula_cache(&[0.0, 0.5, 1.0]);
        let options = DecorOptions {
            alpha: 999.0,
            gamma: 1.0,
            weight_formula: DecorWeightFormula::PowerRetention,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert_close(ctx.penalty(0.5).unwrap(), 0.5);

        let options = DecorOptions {
            gamma: 0.0,
            weight_formula: DecorWeightFormula::PowerRetention,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert_close(ctx.penalty(0.9).unwrap(), 1.0);
    }

    #[test]
    fn decor_formula_validation_rejects_invalid_parameters() {
        for options in [
            DecorOptions {
                gamma: -1.0,
                ..DecorOptions::default()
            },
            DecorOptions {
                threshold_tau: 1.0,
                ..DecorOptions::default()
            },
            DecorOptions {
                penalty_floor: 1.0,
                ..DecorOptions::default()
            },
            DecorOptions {
                scale_epsilon: 0.0,
                ..DecorOptions::default()
            },
        ] {
            assert!(DecorFormulaContext::from_cache(&formula_cache(&[0.1]), &options).is_err());
        }
    }

    #[test]
    fn decor_distribution_formulas_error_without_finite_cache_scores() {
        let cache = formula_cache(&[]);
        let options = DecorOptions {
            weight_formula: DecorWeightFormula::ScaledRational,
            ..DecorOptions::default()
        };
        let err = DecorFormulaContext::from_cache(&cache, &options)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no finite redundancy scores"));
    }

    #[test]
    fn decor_es_matches_classic_when_redundancy_zero() {
        let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 2, 5];
        let penalty = vec![1.0, 1.0, 1.0];
        let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
        assert!((decor - classic).abs() < 1e-12);
    }

    #[test]
    fn decor_alpha_zero_matches_classic() {
        let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 2, 5];
        let cache = formula_cache(&[0.9, 0.1, 0.4]);
        let options = DecorOptions {
            alpha: 0.0,
            weight_formula: DecorWeightFormula::RawRational,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        let penalty = ctx.penalties_for(&[0.9, 0.1, 0.4]).unwrap();
        let decor = calculate_es_decor(&stats, &hits, &penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
        assert!((decor - classic).abs() < 1e-12);
    }

    #[test]
    fn decor_downweights_high_redundancy_gene() {
        let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 1, 5];
        let no_penalty = vec![1.0, 1.0, 1.0];
        let penalized = vec![1.0 / 2.0, 1.0, 1.0];
        let classic = calculate_es_decor(&stats, &hits, &no_penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let decor = calculate_es_decor(&stats, &hits, &penalized, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        assert_ne!(decor, classic);
        assert!(decor.is_finite());
    }

    #[test]
    fn decor_uniform_penalty_cancels_like_classic_weight_normalization() {
        let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 1, 5];
        let uniform_penalty = vec![0.25, 0.25, 0.25];
        let decor =
            calculate_es_decor(&stats, &hits, &uniform_penalty, stats.len(), ScoreType::Std)
                .unwrap()
                .0;
        let classic = calculate_es_fgsea(&stats, &hits, stats.len(), ScoreType::Std).0;
        assert_close(decor, classic);
    }

    #[test]
    fn decor_all_zero_adjusted_weights_fall_back_to_uniform_hit_weights() {
        let stats = vec![5.0, 4.0, 1.0, -1.0, -2.0, -3.0];
        let hits = vec![0, 1, 5];
        let zero_penalty = vec![0.0, 0.0, 0.0];
        let decor = calculate_es_decor(&stats, &hits, &zero_penalty, stats.len(), ScoreType::Std)
            .unwrap()
            .0;
        let uniform_stats = vec![1.0; stats.len()];
        let uniform = calculate_es_fgsea(&uniform_stats, &hits, stats.len(), ScoreType::Std).0;
        assert_close(decor, uniform);
    }

    #[test]
    fn invalid_redundancy_length_errors_cleanly() {
        let err = calculate_es_decor(&[1.0, 2.0], &[0, 1], &[1.0], 2, ScoreType::Std)
            .unwrap_err()
            .to_string();
        assert!(err.contains("length mismatch"));
    }

    fn scalar_counts_for_selected(
        stats: &[f64],
        selected: &[usize],
        penalties: &[Vec<f64>],
        observed_es: &[f64],
        score_type: ScoreType,
    ) -> Vec<DecorNullCounts> {
        let mut expected = vec![DecorNullCounts::default(); penalties.len()];
        for (i, penalty) in penalties.iter().enumerate() {
            let (rand_es, _) =
                calculate_es_decor_prechecked(stats, selected, penalty, stats.len(), score_type);
            update_decor_null_count(&mut expected[i], rand_es, observed_es[i]);
        }
        expected
    }

    fn rank_major_penalties(penalties: &[Vec<f64>]) -> Vec<f64> {
        let size = penalties[0].len();
        let mut out = Vec::with_capacity(size * penalties.len());
        for rank_idx in 0..size {
            for penalty in penalties {
                out.push(penalty[rank_idx]);
            }
        }
        out
    }

    #[test]
    fn decor_batched_counts_match_scalar_es_updates_for_all_score_types() {
        let stats = vec![4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let stats_abs = stats.iter().map(|v: &f64| v.abs()).collect::<Vec<_>>();
        let selected = vec![0, 2, 5];
        let penalties = [vec![1.0, 0.5, 1.0], vec![0.2, 1.0, 0.8]];
        let observed_es = vec![0.4, -0.1];
        let penalties_rank_major = rank_major_penalties(&penalties);

        for score_type in [ScoreType::Std, ScoreType::Pos, ScoreType::Neg] {
            let mut counts = vec![DecorNullCounts::default(); 2];
            let mut scratch = DecorBatchScratch::default();
            update_decor_batched_counts(
                &stats_abs,
                &selected,
                &penalties_rank_major,
                &observed_es,
                stats.len(),
                score_type,
                &mut counts,
                &mut scratch,
            );
            let expected =
                scalar_counts_for_selected(&stats, &selected, &penalties, &observed_es, score_type);
            assert_eq!(counts, expected);
        }
    }

    #[test]
    fn decor_runtime_size_group_forces_batched_layout_at_threshold() {
        let stats = vec![4.0, 3.0, 2.0, 1.0, -1.0, -2.0];
        let hits = vec![0, 2, 5];
        let work = (0..DECOR_BATCH_MIN_GROUP_PATHWAYS)
            .map(|i| {
                let penalty = vec![1.0 / (1.0 + i as f64 * 0.01), 0.8, 0.6 + i as f64 * 0.001];
                let (es, peak_idx) = calculate_es_decor_prechecked(
                    &stats,
                    &hits,
                    &penalty,
                    stats.len(),
                    ScoreType::Std,
                );
                DecorWorking {
                    pathway_name: format!("PW_{i}"),
                    size: hits.len(),
                    hits: hits.clone(),
                    penalty,
                    es,
                    peak_idx,
                    n_le_es: 0,
                    n_ge_es: 0,
                    n_le_zero: 0,
                    n_ge_zero: 0,
                    le_zero_sum: 0.0,
                    ge_zero_sum: 0.0,
                    nes: None,
                    p_value: f64::NAN,
                    padj: None,
                    log2err: None,
                }
            })
            .collect::<Vec<_>>();

        let indices = (0..work.len()).collect::<Vec<_>>();
        let group = DecorRuntimeSizeGroup::from_indices(hits.len(), indices, &work);
        assert!(group.use_batched);
        assert_eq!(
            group.penalties_rank_major.len(),
            hits.len() * DECOR_BATCH_MIN_GROUP_PATHWAYS
        );
        for rank_idx in 0..hits.len() {
            for (work_idx, w) in work.iter().enumerate().take(DECOR_BATCH_MIN_GROUP_PATHWAYS) {
                assert_eq!(
                    group.penalties_rank_major
                        [rank_idx * DECOR_BATCH_MIN_GROUP_PATHWAYS + work_idx],
                    w.penalty[rank_idx]
                );
            }
        }
    }

    #[test]
    fn decor_grouped_null_matches_scalar_reference_for_mixed_sizes() {
        let stats = vec![5.0, 4.0, 3.0, 2.0, -1.0, -2.0, -3.0];
        let stats_abs = stats.iter().map(|v: &f64| v.abs()).collect::<Vec<_>>();
        let selected_by_size = BTreeMap::from([
            (2usize, vec![0usize, 5usize]),
            (3usize, vec![0usize, 3usize, 6usize]),
        ]);
        let mut work = vec![
            test_working("PW2_A", &stats, vec![0, 2], vec![1.0, 0.7]),
            test_working("PW3_A", &stats, vec![0, 3, 6], vec![1.0, 0.8, 0.6]),
            test_working("PW2_B", &stats, vec![1, 5], vec![0.5, 1.0]),
            test_working("PW3_B", &stats, vec![2, 4, 6], vec![0.9, 0.4, 1.0]),
        ];

        let mut size_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (idx, w) in work.iter().enumerate() {
            size_groups.entry(w.size).or_default().push(idx);
        }
        let runtime_groups = size_groups
            .into_iter()
            .map(|(size, indices)| DecorRuntimeSizeGroup::from_indices(size, indices, &work))
            .collect::<Vec<_>>();
        for group in &runtime_groups {
            let selected = selected_by_size.get(&group.size).unwrap();
            let mut counts = vec![DecorNullCounts::default(); group.work_indices.len()];
            if group.use_batched {
                let mut scratch = DecorBatchScratch::default();
                update_decor_batched_counts(
                    &stats_abs,
                    selected,
                    &group.penalties_rank_major,
                    &group.observed_es,
                    stats.len(),
                    ScoreType::Std,
                    &mut counts,
                    &mut scratch,
                );
            } else {
                for (count, work_idx) in counts.iter_mut().zip(group.work_indices.iter()) {
                    let w = &work[*work_idx];
                    let (rand_es, _) = calculate_es_decor_prechecked(
                        &stats,
                        selected,
                        &w.penalty,
                        stats.len(),
                        ScoreType::Std,
                    );
                    update_decor_null_count(count, rand_es, w.es);
                }
            }

            for (&work_idx, count) in group.work_indices.iter().zip(counts) {
                let w = &mut work[work_idx];
                w.n_le_es += count.n_le_es;
                w.n_ge_es += count.n_ge_es;
                w.n_le_zero += count.n_le_zero;
                w.n_ge_zero += count.n_ge_zero;
                w.le_zero_sum += count.le_zero_sum;
                w.ge_zero_sum += count.ge_zero_sum;
            }
        }

        for w in &work {
            let selected = selected_by_size.get(&w.size).unwrap();
            let expected = scalar_counts_for_selected(
                &stats,
                selected,
                std::slice::from_ref(&w.penalty),
                &[w.es],
                ScoreType::Std,
            );
            assert_eq!(
                DecorNullCounts {
                    n_le_es: w.n_le_es,
                    n_ge_es: w.n_ge_es,
                    n_le_zero: w.n_le_zero,
                    n_ge_zero: w.n_ge_zero,
                    le_zero_sum: w.le_zero_sum,
                    ge_zero_sum: w.ge_zero_sum,
                },
                expected[0],
                "{}",
                w.pathway_name
            );
        }
    }

    fn test_working(
        pathway_name: &str,
        stats: &[f64],
        hits: Vec<usize>,
        penalty: Vec<f64>,
    ) -> DecorWorking {
        let (es, peak_idx) =
            calculate_es_decor_prechecked(stats, &hits, &penalty, stats.len(), ScoreType::Std);
        DecorWorking {
            pathway_name: pathway_name.to_string(),
            size: hits.len(),
            hits,
            penalty,
            es,
            peak_idx,
            n_le_es: 0,
            n_ge_es: 0,
            n_le_zero: 0,
            n_ge_zero: 0,
            le_zero_sum: 0.0,
            ge_zero_sum: 0.0,
            nes: None,
            p_value: f64::NAN,
            padj: None,
            log2err: None,
        }
    }

    #[test]
    fn cache_build_write_read_validate_round_trip() {
        let dir = tempdir().unwrap();
        let expr = dir.path().join("expression.tsv");
        fs::write(
            &expr,
            "gene\ts1\ts2\ts3\ts4\nA\t1\t2\t3\t4\nB\t1.1\t2.1\t3.1\t4.1\nC\t4\t3\t2\t1\nE\t1\t1\t1\t1\n",
        )
        .unwrap();
        let gmt = dir.path().join("test.gmt");
        fs::write(&gmt, "PW\tdesc\tA\tB\tC\tE\n").unwrap();
        let expected = DecorCacheExpectedMetadata {
            gmt_sha256: file_sha256(&gmt).unwrap(),
            expression_sha256: Some(file_sha256(&expr).unwrap()),
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            expression_gene_axis: "rows".to_string(),
            expression_has_header: true,
            gene_id_mode: GENE_ID_MODE.to_string(),
        };
        let pathways = vec![Pathway {
            name: "PW".to_string(),
            description: None,
            genes: vec!["A".into(), "B".into(), "C".into(), "E".into()],
        }];
        let cache = build_decor_cache_from_expression(&pathways, &expr, expected.clone()).unwrap();
        let path = dir.path().join("cache.decor.tsv");
        write_decor_cache_atomic(&path, &cache).unwrap();
        let loaded = read_decor_cache(&path).unwrap();
        assert!(validate_decor_cache(&loaded.metadata, &expected).is_compatible());
        let pw = loaded.pathways.get("PW").unwrap();
        assert_eq!(pw.genes, vec!["A", "B", "C", "E"]);
        assert!(pw.redundancy[0] > 0.3);
    }

    #[test]
    fn formula_and_alpha_do_not_affect_cache_compatibility() {
        let cache = formula_cache(&[0.1, 0.2]);
        let expected = DecorCacheExpectedMetadata {
            gmt_sha256: "gmt".to_string(),
            expression_sha256: Some("expr".to_string()),
            correlation: DecorCorrelation::Pearson,
            redundancy: DecorRedundancy::PositiveMean,
            expression_gene_axis: "rows".to_string(),
            expression_has_header: true,
            gene_id_mode: GENE_ID_MODE.to_string(),
        };
        assert!(validate_decor_cache(&cache.metadata, &expected).is_compatible());

        let options = DecorOptions {
            alpha: 0.0,
            weight_formula: DecorWeightFormula::ExpScaled,
            ..DecorOptions::default()
        };
        let ctx = DecorFormulaContext::from_cache(&cache, &options).unwrap();
        assert_eq!(ctx.weight_formula, DecorWeightFormula::ExpScaled);
        assert!(validate_decor_cache(&cache.metadata, &expected).is_compatible());
    }
}
