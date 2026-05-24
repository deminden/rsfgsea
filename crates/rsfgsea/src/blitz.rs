use crate::core::{BlitzOptions, EnrichmentResult, Pathway, RankedList};
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

const LOWESS_ITERS: usize = 3;

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
struct LinearInterp {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LinearInterp {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self { x, y }
    }

    fn at(&self, xq: f64) -> f64 {
        let n = self.x.len();
        if n == 0 {
            return f64::NAN;
        }
        if n == 1 {
            return self.y[0];
        }

        let idx = match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&xq).unwrap())
        {
            Ok(i) => return self.y[i],
            Err(0) => 0,
            Err(i) if i >= n => n - 2,
            Err(i) => i - 1,
        };
        let x0 = self.x[idx];
        let x1 = self.x[idx + 1];
        let y0 = self.y[idx];
        let y1 = self.y[idx + 1];
        if x1 == x0 {
            y0
        } else {
            y0 + (xq - x0) * (y1 - y0) / (x1 - x0)
        }
    }
}

#[derive(Clone)]
struct BlitzModel {
    alpha_pos: LinearInterp,
    beta_pos: LinearInterp,
    pos_ratio: LinearInterp,
    alpha_neg: LinearInterp,
    beta_neg: LinearInterp,
}

#[derive(Clone)]
struct CleanedPathway {
    name: String,
    genes: Vec<String>,
    hit_indices: Vec<usize>,
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
    marker: HitMarker,
    hit_scores: Vec<f64>,
}

impl BlitzScoreScratch {
    fn new(signature_len: usize) -> Self {
        Self {
            marker: HitMarker::new(signature_len),
            hit_scores: Vec::new(),
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

#[derive(Clone)]
struct HitMarker {
    marks: Vec<u32>,
    generation: u32,
}

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

#[derive(Clone)]
struct PythonStringSet {
    table: Vec<Option<(u64, String)>>,
    used: usize,
    fill: usize,
}

impl PythonStringSet {
    fn new() -> Self {
        Self {
            table: vec![None; 8],
            used: 0,
            fill: 0,
        }
    }

    fn from_iter(values: impl IntoIterator<Item = String>) -> Self {
        let mut set = Self::new();
        for value in values {
            set.insert(value);
        }
        set
    }

    fn len(&self) -> usize {
        self.used
    }

    fn members(&self) -> HashSet<String> {
        self.table
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(_, value)| value.clone()))
            .collect()
    }

    fn iter_values(&self) -> impl Iterator<Item = &String> {
        self.table
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(_, value)| value))
    }

    fn intersection_new_set_order(&self, other: &HashSet<String>) -> Vec<String> {
        let mut result = Self::new();
        for value in self.iter_values() {
            if other.contains(value) {
                result.insert(value.clone());
            }
        }
        result.iter_values().cloned().collect()
    }

    fn insert(&mut self, value: String) -> bool {
        let hash = python_ascii_hash_seed0(&value);
        if !self.insert_no_resize(hash, value) {
            return false;
        }
        let mask = self.table.len() - 1;
        if self.fill * 5 >= mask * 3 {
            let min_used = if self.used > 50_000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(min_used);
        }
        true
    }

    fn insert_no_resize(&mut self, hash: u64, value: String) -> bool {
        let mask = self.table.len() - 1;
        let mut slot = hash as usize & mask;
        let mut perturb = hash as usize;
        loop {
            let mut probes = if slot + 9 <= mask { 9 } else { 0 };
            loop {
                match &self.table[slot] {
                    Some((existing_hash, existing_value))
                        if *existing_hash == hash && existing_value == &value =>
                    {
                        return false;
                    }
                    None => {
                        self.table[slot] = Some((hash, value));
                        self.used += 1;
                        self.fill += 1;
                        return true;
                    }
                    _ => {}
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                slot += 1;
            }
            perturb >>= 5;
            slot = (slot * 5 + 1 + perturb) & mask;
        }
    }

    fn resize(&mut self, min_used: usize) {
        let mut new_size = 8usize;
        while new_size <= min_used {
            new_size <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![None; new_size]);
        self.used = 0;
        self.fill = 0;
        for (hash, value) in old.into_iter().flatten() {
            self.insert_no_resize(hash, value);
        }
    }
}

#[derive(Clone)]
struct PythonIntSet {
    table: Vec<Option<usize>>,
    used: usize,
    fill: usize,
}

impl PythonIntSet {
    fn new() -> Self {
        Self {
            table: vec![None; 8],
            used: 0,
            fill: 0,
        }
    }

    fn from_iter(values: impl IntoIterator<Item = usize>) -> Self {
        let mut set = Self::new();
        for value in values {
            set.insert(value);
        }
        set
    }

    fn len(&self) -> usize {
        self.used
    }

    fn contains(&self, value: usize) -> bool {
        let mask = self.table.len() - 1;
        let mut slot = value & mask;
        let mut perturb = value;
        loop {
            let mut probes = if slot + 9 <= mask { 9 } else { 0 };
            loop {
                match self.table[slot] {
                    Some(existing) if existing == value => return true,
                    None => return false,
                    _ => {}
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                slot += 1;
            }
            perturb >>= 5;
            slot = (slot * 5 + 1 + perturb) & mask;
        }
    }

    fn iter_values(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|slot| *slot)
    }

    fn insert(&mut self, value: usize) -> bool {
        if !self.insert_no_resize(value) {
            return false;
        }
        let mask = self.table.len() - 1;
        if self.fill * 5 >= mask * 3 {
            let min_used = if self.used > 50_000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(min_used);
        }
        true
    }

    fn insert_no_resize(&mut self, value: usize) -> bool {
        let mask = self.table.len() - 1;
        let mut slot = value & mask;
        let mut perturb = value;
        loop {
            let mut probes = if slot + 9 <= mask { 9 } else { 0 };
            loop {
                match self.table[slot] {
                    Some(existing) if existing == value => return false,
                    None => {
                        self.table[slot] = Some(value);
                        self.used += 1;
                        self.fill += 1;
                        return true;
                    }
                    _ => {}
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                slot += 1;
            }
            perturb >>= 5;
            slot = (slot * 5 + 1 + perturb) & mask;
        }
    }

    fn resize(&mut self, min_used: usize) {
        let mut new_size = 8usize;
        while new_size <= min_used {
            new_size <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![None; new_size]);
        self.used = 0;
        self.fill = 0;
        for value in old.into_iter().flatten() {
            self.insert_no_resize(value);
        }
    }
}

fn python_ascii_hash_seed0(value: &str) -> u64 {
    let mut hash = siphash13_seed0(value.as_bytes());
    if hash == u64::MAX {
        hash = u64::MAX - 1;
    }
    hash
}

fn siphash13_seed0(data: &[u8]) -> u64 {
    let mut v0 = 0x736f_6d65_7073_6575_u64;
    let mut v1 = 0x646f_7261_6e64_6f6d_u64;
    let mut v2 = 0x6c79_6765_6e65_7261_u64;
    let mut v3 = 0x7465_6462_7974_6573_u64;

    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let m = u64::from_le_bytes(chunk.try_into().unwrap());
        v3 ^= m;
        sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= m;
    }

    let mut b = (data.len() as u64) << 56;
    for (idx, byte) in remainder.iter().enumerate() {
        b |= (*byte as u64) << (8 * idx);
    }
    v3 ^= b;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= b;

    v2 ^= 0xff;
    for _ in 0..3 {
        sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    }
    v0 ^ v1 ^ v2 ^ v3
}

fn sip_round(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = v1.rotate_left(13);
    *v1 ^= *v0;
    *v0 = v0.rotate_left(32);

    *v2 = v2.wrapping_add(*v3);
    *v3 = v3.rotate_left(16);
    *v3 ^= *v2;

    *v0 = v0.wrapping_add(*v3);
    *v3 = v3.rotate_left(21);
    *v3 ^= *v0;

    *v2 = v2.wrapping_add(*v1);
    *v1 = v1.rotate_left(17);
    *v1 ^= *v2;
    *v2 = v2.rotate_left(32);
}

#[derive(Clone)]
struct NumpyMt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl NumpyMt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, mti: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;

        if self.mti >= N {
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            self.mti = 0;
        }

        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn random_interval(&mut self, max: usize) -> usize {
        let mut mask = max as u32;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max as u32 {
                return value as usize;
            }
        }
    }

    #[cfg(test)]
    fn choice_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut values: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.random_interval(i);
            values.swap(i, j);
        }
        values.truncate(k);
        values
    }

    fn choice_without_replacement_into<'a>(
        &mut self,
        n: usize,
        k: usize,
        values: &'a mut Vec<usize>,
    ) -> &'a [usize] {
        values.clear();
        values.extend(0..n);
        for i in (1..n).rev() {
            let j = self.random_interval(i);
            values.swap(i, j);
        }
        values.truncate(k);
        values
    }

    fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    fn standard_normals(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let mut r2 = 2.0;
            let mut x1 = 0.0;
            let mut x2 = 0.0;
            while r2 >= 1.0 || r2 == 0.0 {
                x1 = 2.0 * self.next_f64() - 1.0;
                x2 = 2.0 * self.next_f64() - 1.0;
                r2 = x1 * x1 + x2 * x2;
            }
            let f = (-2.0 * r2.ln() / r2).sqrt();
            out.push(f * x2);
            if out.len() < n {
                out.push(f * x1);
            }
        }
        out
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

    let stage = timings.start();
    let model = estimate_model(&signature, &cleaned, options)?;
    timings.finish("estimate_model", stage);

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
    params_by_size: HashMap<usize, BlitzModelParams>,
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
) -> HashMap<usize, BlitzModelParams> {
    let mut params_by_size = HashMap::new();
    for pathway in cleaned {
        let size = pathway.genes.len();
        if size < options.min_size || size > options.max_size {
            continue;
        }
        params_by_size.entry(size).or_insert_with(|| {
            let size_f = size as f64;
            BlitzModelParams {
                pos_alpha: model.alpha_pos.at(size_f),
                pos_beta: model.beta_pos.at(size_f),
                pos_ratio: model.pos_ratio.at(size_f).clamp(0.0, 1.0),
                neg_alpha: model.alpha_neg.at(size_f),
                neg_beta: model.beta_neg.at(size_f),
            }
        });
    }
    params_by_size
}

fn score_blitz_pathways(
    signature: &BlitzSignature,
    cleaned: &[CleanedPathway],
    params_by_size: &HashMap<usize, BlitzModelParams>,
    options: &BlitzOptions,
) -> Result<Vec<ScoredBlitzPathway>> {
    let processes = options.processes.max(1);
    let score = |scratch: &mut BlitzScoreScratch,
                 pathway: &CleanedPathway|
     -> Result<Option<ScoredBlitzPathway>> {
        let size = pathway.genes.len();
        if size < options.min_size || size > options.max_size {
            return Ok(None);
        }
        let params = params_by_size
            .get(&size)
            .expect("model parameters should be cached for every kept size");
        score_cleaned_pathway(signature, pathway, *params, options.deep_accuracy, scratch).map(Some)
    };

    if processes == 1 {
        let mut scratch = BlitzScoreScratch::new(signature.abs_scores.len());
        let mut out = Vec::new();
        for pathway in cleaned {
            if let Some(row) = score(&mut scratch, pathway)? {
                out.push(row);
            }
        }
        return Ok(out);
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()?;
    pool.install(|| {
        cleaned
            .par_iter()
            .map_init(
                || BlitzScoreScratch::new(signature.abs_scores.len()),
                |scratch, pathway| score(scratch, pathway),
            )
            .collect::<Result<Vec<_>>>()
            .map(|rows| rows.into_iter().flatten().collect())
    })
}

fn score_cleaned_pathway(
    signature: &BlitzSignature,
    pathway: &CleanedPathway,
    params: BlitzModelParams,
    deep_accuracy: usize,
    scratch: &mut BlitzScoreScratch,
) -> Result<ScoredBlitzPathway> {
    let extrema =
        enrichment_score_for_indices(&signature.abs_scores, &pathway.hit_indices, scratch);
    let leading_edge = leading_edge_blitz(
        signature,
        &pathway.leading_hits,
        extrema.rmax,
        extrema.max_value,
        extrema.rmin,
        extrema.min_value,
        extrema.peak_idx,
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
                leading_hits,
            }
        })
        .collect()
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

fn estimate_anchor_fits(
    signature: &BlitzSignature,
    library: &[CleanedPathway],
    options: &BlitzOptions,
) -> Result<(Vec<usize>, Vec<AnchorFit>)> {
    let max_library_size = library
        .iter()
        .map(|pathway| pathway.genes.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let anchor_sizes = anchor_set_sizes(max_library_size, signature.genes.len(), options.anchors);
    if anchor_sizes.is_empty() {
        bail!("blitz calibration produced no valid anchor set sizes.");
    }
    let processes = options.processes.max(1);
    let fits = if processes == 1 {
        let mut rng = NumpyMt19937::new(options.seed as u32);
        anchor_sizes
            .iter()
            .map(|&size| estimate_anchor(signature, size, options, &mut rng))
            .collect::<Result<Vec<_>>>()?
    } else {
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
                jobs.into_iter()
                    .map(|(idx, size)| {
                        estimate_anchor(signature, size, options, rng).map(|fit| (idx, fit))
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
) -> Result<AnchorFit> {
    let mut es = Vec::with_capacity(options.permutations);
    let mut permutation = Vec::with_capacity(signature.abs_scores.len());
    let mut scratch = BlitzScoreScratch::new(signature.abs_scores.len());
    for _ in 0..options.permutations {
        let hits = rng.choice_without_replacement_into(
            signature.abs_scores.len(),
            set_size,
            &mut permutation,
        );
        let value = enrichment_score_for_hits(&signature.abs_scores, hits, &mut scratch.marker);
        if value.is_finite() {
            es.push(value);
        }
    }
    if es.is_empty() {
        bail!("blitz calibration generated no finite enrichment scores for set size {set_size}.");
    }

    let pos = es.iter().copied().filter(|v| *v > 0.0).collect::<Vec<_>>();
    let neg = es.iter().copied().filter(|v| *v < 0.0).collect::<Vec<_>>();
    let mut symmetric = options.symmetric;
    if (neg.len() < 250 || pos.len() < 250) && !symmetric {
        symmetric = true;
    }

    let (pos_fit, neg_fit) = if symmetric {
        let abs = es
            .iter()
            .copied()
            .filter(|v| *v != 0.0)
            .map(f64::abs)
            .collect::<Vec<_>>();
        let fit = fit_gamma_floc0(&abs)?;
        (fit.clone(), fit)
    } else {
        let pos_fit = fit_gamma_floc0(&pos)?;
        let neg_abs = neg.iter().map(|v| -*v).collect::<Vec<_>>();
        let neg_fit = fit_gamma_floc0(&neg_abs)?;
        (pos_fit, neg_fit)
    };

    let denom = pos.len() + neg.len();
    let pos_ratio = if denom == 0 {
        0.5
    } else {
        pos.len() as f64 / denom as f64
    };

    Ok(AnchorFit {
        alpha_pos: pos_fit.alpha,
        beta_pos: pos_fit.beta,
        alpha_neg: neg_fit.alpha,
        beta_neg: neg_fit.beta,
        pos_ratio,
    })
}

fn fit_gamma_floc0(values: &[f64]) -> Result<GammaFit> {
    let clean = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect::<Vec<_>>();
    if clean.is_empty() {
        bail!("cannot fit gamma distribution to an empty positive sample.");
    }
    let clean_f32 = clean.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let log_f32 = clean_f32
        .iter()
        .map(|v| numpy_log_f32(*v))
        .collect::<Vec<_>>();
    let n = clean_f32.len() as f32;
    let mean = numpy_pairwise_sum_f32(&clean_f32) / n;
    let mean_log = numpy_pairwise_sum_f32(&log_f32) / n;
    let s = numpy_log_f32(mean) - mean_log;
    let estimate = (3.0_f32 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s);
    let alpha =
        solve_gamma_shape_brentq((estimate * 0.6) as f64, (estimate * 1.4) as f64, s as f64);
    Ok(GammaFit {
        alpha,
        beta: (mean / alpha as f32) as f64,
    })
}

#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn numpy_log_f32(x_in: f32) -> f32 {
    const P0: f32 = 0.000_000_000_000_000_000_000e0_f32;
    const P1: f32 = 9.999_999_999_999_999e-1_f32;
    const P2: f32 = 2.112_677_543_073_053_f32;
    const P3: f32 = 1.480_000_633_576_506_6_f32;
    const P4: f32 = 3.808_837_741_388_408e-1_f32;
    const P5: f32 = 2.589_979_117_907_922_7e-2_f32;
    const Q0: f32 = 1.000_000_000_000_000_000_000e0_f32;
    const Q1: f32 = 2.612_677_543_073_109_f32;
    const Q2: f32 = 2.453_006_071_784_736_4_f32;
    const Q3: f32 = 9.864_942_958_519_419e-1_f32;
    const Q4: f32 = 1.546_476_374_983_906_7e-1_f32;
    const Q5: f32 = 5.875_095_403_124_574e-3_f32;
    const LOGE2: f32 = 0.693_147_180_559_945_3_f32;
    const SQRT1_2: f32 = 0.707_106_781_186_547_6_f32;

    if x_in.is_nan() {
        return f32::NAN;
    }
    if x_in == f32::INFINITY {
        return f32::INFINITY;
    }
    if x_in == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x_in < 0.0 {
        return -f32::NAN;
    }
    if x_in < f32::MIN_POSITIVE {
        return x_in.ln();
    }

    let bits = x_in.to_bits();
    let mut exponent = (((bits >> 23) & 0xff) as i32 - 0x7e) as f32;
    let mut x = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);

    if x <= SQRT1_2 {
        x += x;
        exponent -= 1.0;
    }
    x -= 1.0;

    let mut num_poly = P5.mul_add(x, P4);
    num_poly = num_poly.mul_add(x, P3);
    num_poly = num_poly.mul_add(x, P2);
    num_poly = num_poly.mul_add(x, P1);
    num_poly = num_poly.mul_add(x, P0);

    let mut denom_poly = Q5.mul_add(x, Q4);
    denom_poly = denom_poly.mul_add(x, Q3);
    denom_poly = denom_poly.mul_add(x, Q2);
    denom_poly = denom_poly.mul_add(x, Q1);
    denom_poly = denom_poly.mul_add(x, Q0);

    exponent.mul_add(LOGE2, num_poly / denom_poly)
}

fn numpy_pairwise_sum_f32(values: &[f32]) -> f32 {
    const PW_BLOCKSIZE: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut res = -0.0_f32;
        for value in values {
            res += *value;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut i = 8;
        let limit = n - (n % 8);
        while i < limit {
            r[0] += values[i];
            r[1] += values[i + 1];
            r[2] += values[i + 2];
            r[3] += values[i + 3];
            r[4] += values[i + 4];
            r[5] += values[i + 5];
            r[6] += values[i + 6];
            r[7] += values[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_pairwise_sum_f32(&values[..n2]) + numpy_pairwise_sum_f32(&values[n2..])
    }
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
    scratch: &mut BlitzScoreScratch,
) -> EnrichmentExtrema {
    scratch.marker.mark_hits(hits);
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
    for (i, &abs_score) in abs_scores.iter().enumerate() {
        let hit_indicator = if scratch.marker.contains(i) { 1.0 } else { 0.0 };
        csum += hit_indicator * abs_score * norm_hit - (1.0 - hit_indicator) * norm_no_hit;
        if csum >= max_value {
            max_value = csum;
            rmax = i;
        }
        if csum <= min_value {
            min_value = csum;
            rmin = i;
        }
        let cur_abs = csum.abs();
        if cur_abs > best_abs {
            best_abs = cur_abs;
            best_idx = i;
            best_value = csum;
        }
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

fn enrichment_score_for_hits(abs_scores: &[f64], hits: &[usize], marker: &mut HitMarker) -> f64 {
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

fn leading_edge_blitz(
    signature: &BlitzSignature,
    hits: &PythonIntSet,
    rmax: usize,
    max_value: f64,
    rmin: usize,
    min_value: f64,
    _peak_idx: usize,
) -> Vec<String> {
    let running_len = signature.abs_scores.len();
    if running_len == 0 {
        return Vec::new();
    }
    let idxs = if max_value > min_value.abs() {
        if rmax < hits.len() {
            let filtered = PythonIntSet::from_iter(0..rmax)
                .iter_values()
                .filter(|&idx| hits.contains(idx))
                .collect::<Vec<_>>();
            PythonIntSet::from_iter(filtered)
                .iter_values()
                .collect::<Vec<_>>()
        } else {
            let filtered = hits
                .iter_values()
                .filter(|&idx| idx < rmax)
                .collect::<Vec<_>>();
            PythonIntSet::from_iter(filtered)
                .iter_values()
                .collect::<Vec<_>>()
        }
    } else {
        let range_len = running_len.saturating_sub(rmin);
        if range_len < hits.len() {
            let filtered = PythonIntSet::from_iter(rmin..running_len)
                .iter_values()
                .filter(|&idx| hits.contains(idx))
                .collect::<Vec<_>>();
            PythonIntSet::from_iter(filtered)
                .iter_values()
                .collect::<Vec<_>>()
        } else {
            let filtered = hits
                .iter_values()
                .filter(|&idx| idx >= rmin && idx < running_len)
                .collect::<Vec<_>>();
            PythonIntSet::from_iter(filtered)
                .iter_values()
                .collect::<Vec<_>>()
        }
    };
    idxs.into_iter()
        .map(|idx| signature.genes[idx].clone())
        .collect()
}

#[cfg(test)]
fn python_int_set_iteration_order(values: &[usize]) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let mut table_size = 8usize;
    while unique.len() * 5 >= (table_size - 1) * 3 {
        table_size *= 4;
    }
    let mask = table_size - 1;
    let mut table = vec![None; table_size];

    for value in unique {
        let mut slot = value & mask;
        let mut perturb = value;
        loop {
            let mut probes = if slot + 9 <= mask { 9 } else { 0 };
            loop {
                if table[slot].is_none() {
                    table[slot] = Some(value);
                    break;
                }
                if probes == 0 {
                    break;
                }
                probes -= 1;
                slot += 1;
            }
            if table[slot] == Some(value) {
                break;
            }
            perturb >>= 5;
            slot = (slot * 5 + 1 + perturb) & mask;
        }
    }

    table.into_iter().flatten().collect()
}

fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if !x.is_finite() || !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return f64::NAN;
    }
    crate::blitz_gamma::scipy_1_16_3_gammainc(alpha, x / beta)
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

fn lowess_interpolation(x: &[f64], y: &[f64], frac: f64) -> LinearInterp {
    LinearInterp::new(x.to_vec(), lowess(y, x, frac))
}

fn lowess(y: &[f64], x: &[f64], frac: f64) -> Vec<f64> {
    let n = x.len();
    if n <= 2 {
        return y.to_vec();
    }
    let k = ((frac * n as f64 + 1e-10) as usize).clamp(2, n);
    let mut residual_weights = vec![1.0; n];
    let mut fitted = vec![0.0; n];

    for iter in 0..=LOWESS_ITERS {
        fitted.fill(0.0);
        let mut left_end = 0usize;
        let mut right_end = k;

        for i in 0..n {
            let xval = x[i];
            while right_end < n && xval > (x[left_end] + x[right_end]) / 2.0 {
                left_end += 1;
                right_end += 1;
            }
            let radius = (xval - x[left_end]).max(x[right_end - 1] - xval);

            let mut weights = vec![0.0; n];
            let mut nonzero_weights = 0usize;
            for j in left_end..right_end {
                let dist = ((x[j] - xval).abs() / radius).clamp(0.0, 1.0);
                let dist3 = dist * dist * dist;
                let tricube = 1.0 - dist3;
                let w = (tricube * tricube * tricube) * residual_weights[j];
                weights[j] = w;
                if w > 1e-12 {
                    nonzero_weights += 1;
                }
            }
            let sum_weights = numpy_pairwise_sum_f64(&weights[left_end..right_end]);

            if nonzero_weights < 2 || sum_weights <= 0.0 {
                fitted[i] = y[i];
                continue;
            }

            for weight in &mut weights[left_end..right_end] {
                *weight /= sum_weights;
            }
            let sum_weighted_x = (left_end..right_end)
                .map(|j| weights[j] * x[j])
                .sum::<f64>();
            let weighted_sqdev_x = (left_end..right_end)
                .map(|j| weights[j] * (x[j] - sum_weighted_x).powf(2.0))
                .sum::<f64>()
                .max(1e-12);
            fitted[i] = (left_end..right_end)
                .map(|j| {
                    let projection = weights[j]
                        * (1.0
                            + (xval - sum_weighted_x) * (x[j] - sum_weighted_x) / weighted_sqdev_x);
                    projection * y[j]
                })
                .sum();
        }

        if iter == LOWESS_ITERS {
            break;
        }

        let mut residuals = y
            .iter()
            .zip(&fitted)
            .map(|(yi, fi)| (yi - fi).abs())
            .collect::<Vec<_>>();
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n.is_multiple_of(2) {
            0.5 * (residuals[n / 2 - 1] + residuals[n / 2])
        } else {
            residuals[n / 2]
        };
        if median == 0.0 {
            for i in 0..n {
                residual_weights[i] = if (y[i] - fitted[i]).abs() > 0.0 {
                    0.0
                } else {
                    1.0
                };
            }
        } else {
            let scale = 6.0 * median;
            for i in 0..n {
                let u = ((y[i] - fitted[i]).abs() / scale).min(1.0);
                let bisquare = 1.0 - u * u;
                residual_weights[i] = bisquare * bisquare;
            }
        }
    }
    fitted
}

fn numpy_pairwise_sum_f64(values: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = values.len();
    if n < 8 {
        let mut res = -0.0_f64;
        for value in values {
            res += *value;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut i = 8;
        let limit = n - (n % 8);
        while i < limit {
            r[0] += values[i];
            r[1] += values[i + 1];
            r[2] += values[i + 2];
            r[3] += values[i + 3];
            r[4] += values[i + 4];
            r[5] += values[i + 5];
            r[6] += values[i + 6];
            r[7] += values[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_pairwise_sum_f64(&values[..n2]) + numpy_pairwise_sum_f64(&values[n2..])
    }
}

fn numpy_hit_score_sum_f64(values: &[f64]) -> f64 {
    if values.len() == 7 {
        (values[0] + (values[1] + values[2])) + ((values[3] + values[4]) + (values[5] + values[6]))
    } else {
        numpy_pairwise_sum_f64(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{read_gmt, read_ranked_list};
    use serde::Deserialize;

    #[test]
    fn numpy_mt19937_permutation_matches_reference() {
        let cases = [
            (0, vec![2, 8, 4, 9, 1]),
            (1, vec![2, 9, 6, 4, 0]),
            (42, vec![8, 1, 5, 0, 7]),
            (20_260_322, vec![3, 7, 9, 0, 4]),
        ];
        for (seed, expected) in cases {
            let mut rng = NumpyMt19937::new(seed);
            assert_eq!(&rng.choice_without_replacement(10, 5), &expected);
        }
    }

    #[test]
    fn numpy_mt19937_standard_normals_match_reference() {
        let mut rng = NumpyMt19937::new(0);
        let observed = rng.standard_normals(10);
        let expected = [
            1.764052345967664,
            0.4001572083672233,
            0.9787379841057392,
            2.240893199201458,
            1.8675579901499675,
            -0.977277879876411,
            0.9500884175255894,
            -0.1513572082976979,
            -0.10321885179355784,
            0.41059850193837233,
        ];
        for (observed, expected) in observed.iter().zip(expected) {
            assert!((observed - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn python_int_set_iteration_matches_reference_cases() {
        assert_eq!(python_int_set_iteration_order(&[2, 9]), vec![9, 2]);
        assert_eq!(
            python_int_set_iteration_order(&(105..120).collect::<Vec<_>>()),
            (105..120).collect::<Vec<_>>()
        );
        assert_eq!(
            python_int_set_iteration_order(&[0, 16, 32, 48, 64]),
            vec![0, 32, 64, 16, 48]
        );
    }

    #[test]
    fn python_string_hash_and_set_order_match_seed0_reference() {
        assert_eq!(python_ascii_hash_seed0("g001") as i64, 3270876322613014562);
        assert_eq!(python_ascii_hash_seed0("g004") as i64, -8038800456378607197);
        let set = PythonStringSet::from_iter(
            [
                "g001", "g004", "g009", "g008", "g007", "g003", "g012", "g010", "g002", "g011",
                "g006", "g005",
            ]
            .into_iter()
            .map(str::to_string),
        );
        assert_eq!(
            set.iter_values().cloned().collect::<Vec<_>>(),
            [
                "g001", "g004", "g008", "g007", "g003", "g012", "g010", "g002", "g006", "g011",
                "g009", "g005",
            ]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn blitz_mode_runs_tiny_fixture() {
        let ranks = RankedList::new(
            vec!["g1", "g2", "g3", "g4", "g5", "g6"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            vec![3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
        );
        let pathways = vec![
            Pathway {
                name: "PW_A".to_string(),
                description: None,
                genes: vec!["g1", "g2", "g3"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            },
            Pathway {
                name: "PW_B".to_string(),
                description: None,
                genes: vec!["g4", "g5", "g6"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            },
        ];
        let options = BlitzOptions {
            permutations: 64,
            anchors: 8,
            min_size: 1,
            max_size: 6,
            processes: 1,
            ..BlitzOptions::default()
        };
        let res = fgsea_blitz_with_options(&ranks, &pathways, &options).unwrap();
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|row| row.p_value.is_finite()));
    }

    #[test]
    fn blitz_prepare_signature_sorts_before_deduplicating() {
        let ranks = RankedList {
            genes: ["dup", "low", "dup", "high", "mid"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            scores: vec![-2.0, -5.0, 4.0, 8.0, 1.0],
        };
        let signature = prepare_signature(&ranks, false);
        assert_eq!(
            signature.genes,
            ["high", "dup", "mid", "low"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        );
        assert_eq!(signature.abs_scores, vec![8.0, 4.0, 1.0, 5.0]);
    }

    #[test]
    fn blitz_clean_pathways_matches_python_string_set_order_reference() {
        let (_, cleaned) = reference_inputs();
        let top = cleaned
            .iter()
            .find(|pathway| pathway.name == "TOP_12")
            .map(|pathway| &pathway.genes)
            .unwrap();
        assert_eq!(
            top,
            &[
                "g001", "g004", "g009", "g008", "g007", "g003", "g010", "g002", "g006", "g011",
                "g012", "g005",
            ]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn optimized_enrichment_score_extrema_match_running_sum_reference() {
        let abs_scores = vec![3.0, 1.5, 7.0, 2.25, 4.0, 5.5, 0.75, 6.25, 3.75];
        let hits = vec![7, 0, 4, 2];
        let mut scratch = BlitzScoreScratch::new(abs_scores.len());
        let observed = enrichment_score_for_indices(&abs_scores, &hits, &mut scratch);

        let mut hit_indicator = vec![0.0; abs_scores.len()];
        for &hit in &hits {
            hit_indicator[hit] = 1.0;
        }
        let number_hits = hits.len();
        let number_miss = abs_scores.len().saturating_sub(number_hits);
        let hit_scores = hits.iter().map(|&idx| abs_scores[idx]).collect::<Vec<_>>();
        let sum_hit_scores = numpy_hit_score_sum_f64(&hit_scores);
        let norm_hit = 1.0 / sum_hit_scores;
        let norm_no_hit = 1.0 / number_miss as f64;
        let mut running = Vec::new();
        let mut csum = 0.0;
        let mut best_idx = 0usize;
        let mut best_abs = f64::NEG_INFINITY;
        for i in 0..abs_scores.len() {
            csum += hit_indicator[i] * abs_scores[i] * norm_hit
                - (1.0 - hit_indicator[i]) * norm_no_hit;
            running.push(csum);
            let cur_abs = csum.abs();
            if cur_abs > best_abs {
                best_abs = cur_abs;
                best_idx = i;
            }
        }
        let (rmax, max_value) = running
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();
        let (rmin, min_value) = running
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();

        assert_f64_bits("optimized ES", observed.es, running[best_idx]);
        assert_eq!(observed.peak_idx, best_idx);
        assert_eq!(observed.rmax, rmax);
        assert_eq!(observed.rmin, rmin);
        assert_f64_bits("optimized max", observed.max_value, max_value);
        assert_f64_bits("optimized min", observed.min_value, min_value);
    }

    #[test]
    fn parallel_final_scoring_matches_sequential_scoring_exactly() {
        let (signature, cleaned) = reference_inputs();
        let options = BlitzOptions::default();
        let model = estimate_model(&signature, &cleaned, &options).unwrap();
        let params = model_params_by_size(&model, &cleaned, &options);
        let mut sequential_options = options.clone();
        sequential_options.processes = 1;
        let mut parallel_options = options.clone();
        parallel_options.processes = 4;

        let sequential =
            score_blitz_pathways(&signature, &cleaned, &params, &sequential_options).unwrap();
        let parallel =
            score_blitz_pathways(&signature, &cleaned, &params, &parallel_options).unwrap();
        assert_eq!(sequential.len(), parallel.len());
        for (seq, par) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(seq.result.pathway_name, par.result.pathway_name);
            assert_eq!(seq.result.size, par.result.size);
            assert_eq!(seq.result.leading_edge, par.result.leading_edge);
            assert_f64_bits(
                &format!("{} ES", seq.result.pathway_name),
                seq.result.es,
                par.result.es,
            );
            assert_f64_bits(
                &format!("{} pval", seq.result.pathway_name),
                seq.result.p_value,
                par.result.p_value,
            );
            assert_f64_bits(
                &format!("{} NES", seq.result.pathway_name),
                seq.result.nes.unwrap(),
                par.result.nes.unwrap(),
            );
        }
    }

    #[derive(Debug, Deserialize)]
    struct AnchorTraceRow {
        set_size: usize,
        alpha_pos: f64,
        beta_pos: f64,
        alpha_neg: f64,
        beta_neg: f64,
        pos_ratio: f64,
        alpha_pos_smooth: f64,
        beta_pos_smooth: f64,
        alpha_neg_smooth: f64,
        beta_neg_smooth: f64,
        pos_ratio_jittered: f64,
        pos_ratio_smooth: f64,
    }

    #[derive(Debug, Deserialize)]
    struct ResultTraceRow {
        pathway: String,
        set_size: usize,
        es: f64,
        pos_alpha: f64,
        pos_beta: f64,
        pos_ratio_clipped: f64,
        neg_alpha: f64,
        neg_beta: f64,
        nes: f64,
        pval: f64,
    }

    #[derive(Debug, Deserialize)]
    struct SignatureTraceRow {
        gene: String,
        centered_score: f64,
        abs_score: f64,
    }

    #[derive(Debug, Deserialize)]
    struct TailTraceRow {
        case: String,
        branch: String,
        x: f64,
        alpha: f64,
        beta: f64,
        pos_ratio: f64,
        deep_accuracy: usize,
        fallback_used: bool,
        gamma_prob: f64,
        survival_prob: f64,
        prob_two_tailed: f64,
        pval: f64,
        nes: f64,
    }

    fn reference_inputs() -> (BlitzSignature, Vec<CleanedPathway>) {
        let root = env!("CARGO_MANIFEST_DIR");
        let ranks =
            read_ranked_list(format!("{root}/tests/data/blitz_reference/synthetic.rnk")).unwrap();
        let pathways = read_gmt(format!("{root}/tests/data/blitz_reference/synthetic.gmt"))
            .unwrap()
            .pathways;
        let signature = prepare_signature(&ranks, true);
        let cleaned = clean_pathways(&pathways, &signature);
        (signature, cleaned)
    }

    fn read_anchor_trace() -> Vec<AnchorTraceRow> {
        let root = env!("CARGO_MANIFEST_DIR");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(format!(
                "{root}/tests/data/blitz_reference/synthetic.trace_anchors.tsv"
            ))
            .unwrap();
        reader
            .deserialize::<AnchorTraceRow>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn read_signature_trace() -> Vec<SignatureTraceRow> {
        let root = env!("CARGO_MANIFEST_DIR");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(format!(
                "{root}/tests/data/blitz_reference/synthetic.trace_signature.tsv"
            ))
            .unwrap();
        reader
            .deserialize::<SignatureTraceRow>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn read_result_trace() -> Vec<ResultTraceRow> {
        let root = env!("CARGO_MANIFEST_DIR");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(format!(
                "{root}/tests/data/blitz_reference/synthetic.trace_results.tsv"
            ))
            .unwrap();
        reader
            .deserialize::<ResultTraceRow>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn read_tail_trace() -> Vec<TailTraceRow> {
        let root = env!("CARGO_MANIFEST_DIR");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(format!(
                "{root}/tests/data/blitz_reference/tail_fallback.trace_gamma.tsv"
            ))
            .unwrap();
        reader
            .deserialize::<TailTraceRow>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn assert_f64_bits(label: &str, observed: f64, expected: f64) {
        assert_eq!(
            observed.to_bits(),
            expected.to_bits(),
            "{label}: observed {observed:?}, expected {expected:?}"
        );
    }

    #[test]
    fn blitz_anchor_trace_matches_reference() {
        let (signature, cleaned) = reference_inputs();
        let options = BlitzOptions::default();
        let signature_trace = read_signature_trace();
        assert_eq!(signature.genes.len(), signature_trace.len());
        for (idx, expected) in signature_trace.iter().enumerate() {
            assert_eq!(signature.genes[idx], expected.gene);
            assert_f64_bits(
                &format!("{} abs(centered score)", expected.gene),
                signature.abs_scores[idx],
                expected.centered_score.abs(),
            );
            assert_f64_bits(
                &format!("{} abs score", expected.gene),
                signature.abs_scores[idx],
                expected.abs_score,
            );
        }
        let (anchor_sizes, fits) = estimate_anchor_fits(&signature, &cleaned, &options).unwrap();
        let trace = read_anchor_trace();
        assert_eq!(anchor_sizes.len(), trace.len());
        assert_eq!(fits.len(), trace.len());

        let x = anchor_sizes.iter().map(|&v| v as f64).collect::<Vec<_>>();
        let alpha_pos = fits.iter().map(|fit| fit.alpha_pos).collect::<Vec<_>>();
        let beta_pos = fits.iter().map(|fit| fit.beta_pos).collect::<Vec<_>>();
        let alpha_neg = fits.iter().map(|fit| fit.alpha_neg).collect::<Vec<_>>();
        let beta_neg = fits.iter().map(|fit| fit.beta_neg).collect::<Vec<_>>();
        let mut jitter_rng = NumpyMt19937::new(options.seed as u32);
        let jitters = jitter_rng.standard_normals(fits.len());
        let pos_ratio_jittered = fits
            .iter()
            .zip(jitters)
            .map(|(fit, jitter)| (fit.pos_ratio - (0.0001 * jitter).abs()).clamp(0.0, 1.0))
            .collect::<Vec<_>>();
        let alpha_pos_smooth = lowess(&alpha_pos, &x, 0.6);
        let beta_pos_smooth = lowess(&beta_pos, &x, 0.15);
        let alpha_neg_smooth = lowess(&alpha_neg, &x, 0.6);
        let beta_neg_smooth = lowess(&beta_neg, &x, 0.15);
        let pos_ratio_smooth = lowess(&pos_ratio_jittered, &x, 0.5);

        for (idx, ((&set_size, fit), expected)) in
            anchor_sizes.iter().zip(&fits).zip(&trace).enumerate()
        {
            assert_eq!(set_size, expected.set_size);
            assert_f64_bits(
                &format!("{set_size} alpha_pos"),
                fit.alpha_pos,
                expected.alpha_pos,
            );
            assert_f64_bits(
                &format!("{set_size} beta_pos"),
                fit.beta_pos,
                expected.beta_pos,
            );
            assert_f64_bits(
                &format!("{set_size} alpha_neg"),
                fit.alpha_neg,
                expected.alpha_neg,
            );
            assert_f64_bits(
                &format!("{set_size} beta_neg"),
                fit.beta_neg,
                expected.beta_neg,
            );
            assert_f64_bits(
                &format!("{set_size} pos_ratio"),
                fit.pos_ratio,
                expected.pos_ratio,
            );
            assert_f64_bits(
                &format!("{set_size} pos_ratio_jittered"),
                pos_ratio_jittered[idx],
                expected.pos_ratio_jittered,
            );
            assert_f64_bits(
                &format!("{set_size} alpha_pos_smooth"),
                alpha_pos_smooth[idx],
                expected.alpha_pos_smooth,
            );
            assert_f64_bits(
                &format!("{set_size} beta_pos_smooth"),
                beta_pos_smooth[idx],
                expected.beta_pos_smooth,
            );
            assert_f64_bits(
                &format!("{set_size} alpha_neg_smooth"),
                alpha_neg_smooth[idx],
                expected.alpha_neg_smooth,
            );
            assert_f64_bits(
                &format!("{set_size} beta_neg_smooth"),
                beta_neg_smooth[idx],
                expected.beta_neg_smooth,
            );
            assert_f64_bits(
                &format!("{set_size} pos_ratio_smooth"),
                pos_ratio_smooth[idx],
                expected.pos_ratio_smooth,
            );
        }
    }

    #[test]
    fn blitz_result_trace_matches_reference() {
        for row in read_result_trace() {
            let (prob_two_tailed, nes) = if row.es > 0.0 {
                let gamma_prob = gamma_cdf(row.es, row.pos_alpha, row.pos_beta);
                let combined =
                    (gamma_prob * row.pos_ratio_clipped + 1.0 - row.pos_ratio_clipped).min(1.0);
                let prob_two_tailed = (1.0 - combined).min(0.5);
                (prob_two_tailed, normal_isf(prob_two_tailed))
            } else {
                let gamma_prob = gamma_cdf(-row.es, row.neg_alpha, row.neg_beta);
                let combined = (gamma_prob - (gamma_prob * row.pos_ratio_clipped)
                    + row.pos_ratio_clipped)
                    .min(1.0);
                let mut prob_two_tailed = (1.0 - combined).min(0.5);
                if prob_two_tailed == 0.5 {
                    prob_two_tailed -= gamma_prob;
                }
                (prob_two_tailed, -normal_isf(prob_two_tailed))
            };
            assert_f64_bits(&format!("{} nes", row.pathway), nes, row.nes);
            assert_f64_bits(
                &format!("{} pval", row.pathway),
                2.0 * prob_two_tailed,
                row.pval,
            );
            assert!(row.set_size > 0);
        }
    }

    #[test]
    fn mpmath_gammacdf_matches_tail_trace_exact_bits() {
        let rows = read_tail_trace();
        assert!(rows.iter().any(|row| row.fallback_used));
        let sampled = rows
            .into_iter()
            .enumerate()
            .filter_map(|(idx, row)| {
                (idx % 19 == 0
                    || row.case == "pos_lower_half"
                    || row.case == "pos_upper_integer"
                    || row.case == "pos_lower_noninteger"
                    || row.case == "neg_lower_integer"
                    || row.case == "neg_upper_integer"
                    || row.case == "neg_upper_noninteger")
                    .then_some(row)
            })
            .collect::<Vec<_>>();
        assert!(sampled.len() >= 10);
        for row in sampled {
            let observed =
                crate::blitz_mpmath::gammacdf(row.x, row.alpha, row.beta, row.deep_accuracy)
                    .unwrap();
            assert_f64_bits(
                &format!("{} gamma_prob", row.case),
                observed.cdf,
                row.gamma_prob,
            );
            assert_f64_bits(
                &format!("{} survival_prob", row.case),
                observed.survival,
                row.survival_prob,
            );
        }
    }

    #[test]
    fn blitz_tail_probability_matches_trace_exact_bits() {
        for row in read_tail_trace() {
            let branch = match row.branch.as_str() {
                "pos" => crate::blitz_mpmath::TailBranch::Positive,
                "neg" => crate::blitz_mpmath::TailBranch::Negative,
                other => panic!("unknown tail trace branch {other}"),
            };
            let observed = crate::blitz_mpmath::tail_probability(
                branch,
                row.x,
                row.alpha,
                row.beta,
                row.pos_ratio,
                row.deep_accuracy,
            )
            .unwrap();
            assert_f64_bits(
                &format!("{} gamma_prob", row.case),
                observed.gamma_prob,
                row.gamma_prob,
            );
            if row.pval == 0.0 && !row.nes.is_finite() {
                assert!(
                    observed.survival_prob.is_finite() && observed.survival_prob >= 0.0,
                    "{} hidden survival_prob should stay finite for underflow sentinel",
                    row.case
                );
            } else {
                assert_f64_bits(
                    &format!("{} survival_prob", row.case),
                    observed.survival_prob,
                    row.survival_prob,
                );
            }
            assert_f64_bits(
                &format!("{} prob_two_tailed", row.case),
                observed.prob_two_tailed,
                row.prob_two_tailed,
            );
            assert_f64_bits(&format!("{} pval", row.case), observed.p_value, row.pval);
            let nes = match branch {
                crate::blitz_mpmath::TailBranch::Positive => normal_isf(observed.prob_two_tailed),
                crate::blitz_mpmath::TailBranch::Negative => {
                    let mut nes = -normal_isf(observed.prob_two_tailed);
                    if nes == 0.0 {
                        nes = -0.0;
                    }
                    nes
                }
            };
            assert_f64_bits(&format!("{} nes", row.case), nes, row.nes);
        }
    }

    #[test]
    fn gamma_cdf_blitz_uses_fallback_without_error() {
        let rows = read_tail_trace()
            .into_iter()
            .filter(|row| {
                row.fallback_used
                    && (row.case == "pos_lower_half"
                        || row.case == "pos_upper_integer"
                        || row.case == "neg_upper_noninteger")
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 3);
        for row in rows {
            let observed = gamma_cdf_blitz(row.x, row.alpha, row.beta, row.deep_accuracy).unwrap();
            assert_f64_bits(
                &format!("{} fallback cdf", row.case),
                observed,
                row.gamma_prob,
            );
        }
    }
}
