#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(unused_assignments)]

use crate::core::ScoreType;
use crate::rng_compat::{Mt19937Compat, combination, uid_wrapper};
use core::range::Range;
use rayon::prelude::*;
use std::collections::BTreeMap;

const EPS: f64 = 1e-13;

struct SegmentTreeI64 {
    t: Vec<i64>,
    b: Vec<i64>,
    k: usize,
    k2: usize,
    log_k: usize,
    block_mask: usize,
}

impl SegmentTreeI64 {
    #[inline(always)]
    fn new(n_: usize) -> Self {
        let mut k = 1usize;
        let mut log_k = 0usize;
        while k * k < n_ {
            k <<= 1;
            log_k += 1;
        }
        let k2 = (n_ - 1) / k + 1;
        let n = k * k;
        Self {
            t: vec![0; n],
            b: vec![0; k2],
            k,
            k2,
            log_k,
            block_mask: k - 1,
        }
    }

    #[inline(always)]
    fn inc(&mut self, mut p: usize, delta: i64) {
        let block_end = p - (p & self.block_mask) + self.block_mask + 1;
        while p < block_end {
            self.t[p] += delta;
            p += 1;
        }
        let mut p1 = p >> self.log_k;
        while p1 < self.k2 {
            self.b[p1] += delta;
            p1 += 1;
        }
    }

    #[inline(always)]
    fn query_r(&self, r: usize) -> i64 {
        if r == 0 {
            return 0;
        }
        let r = r - 1;
        self.t[r] + self.b[r >> self.log_k]
    }
}

struct SegmentTreeF64 {
    t: Vec<f64>,
    b: Vec<f64>,
    k2: usize,
    log_k: usize,
    block_mask: usize,
}

impl SegmentTreeF64 {
    #[inline(always)]
    fn new(n_: usize) -> Self {
        let mut k = 1usize;
        let mut log_k = 0usize;
        while k * k < n_ {
            k <<= 1;
            log_k += 1;
        }
        let k2 = (n_ - 1) / k + 1;
        let n = k * k;
        Self {
            t: vec![0.0; n],
            b: vec![0.0; k2],
            k2,
            log_k,
            block_mask: k - 1,
        }
    }

    #[inline(always)]
    fn inc(&mut self, mut p: usize, delta: f64) {
        let block_end = p - (p & self.block_mask) + self.block_mask + 1;
        while p < block_end {
            self.t[p] += delta;
            p += 1;
        }
        let mut p1 = p >> self.log_k;
        while p1 < self.k2 {
            self.b[p1] += delta;
            p1 += 1;
        }
    }

    #[inline(always)]
    fn query_r(&self, r: usize) -> f64 {
        if r == 0 {
            return 0.0;
        }
        let r = r - 1;
        self.t[r] + self.b[r >> self.log_k]
    }
}

fn order(x: &[usize]) -> Vec<usize> {
    let mut res: Vec<usize> = (0..x.len()).collect();
    res.sort_by_key(|&i| x[i]);
    res
}

// Fast path when the underlying stats array is already sorted ascending
// (true for fgsea's prepareStats output). In that case selected indices
// themselves determine order, so we just sort them directly.
#[inline]
fn order_sorted_stats(x: &[usize]) -> Vec<usize> {
    let mut res = x.to_vec();
    res.sort_unstable();
    // Now res holds the selected indices in ascending order.
    // We need the permutation that maps position->original-index, i.e.
    // selected_order[i] = j means x[j] is the i-th smallest.
    // Since x is already sorted ascending by value, sorting x directly
    // gives us the indices in value-order.  But we need the inverse
    // permutation.  Build it:
    let mut selected_order = vec![0usize; x.len()];
    for (sorted_pos, &orig_val) in res.iter().enumerate() {
        // orig_val is the stat-position; we need to find which element
        // of the original `x` had this value.
        // Actually we need: selected_order[i] = index j in original x
        // such that x[j] is the i-th smallest.
        // Since x is already sorted by underlying stats, smaller x[j]
        // means smaller stat.  So sorting x's values gives the order.
        // But x holds positions, not values.  We need to map back.
        // Simpler: just sort indices by x[i] using the indices themselves.
        // (This path is only reached when stats are monotone, so x[i]
        //  comparison == stat comparison.)
        let mut j = 0usize;
        while j < x.len() && x[j] != orig_val {
            j += 1;
        }
        selected_order[sorted_pos] = j;
    }
    selected_order
}

fn ranks_from_order(ord: &[usize]) -> Vec<usize> {
    let mut res = vec![0usize; ord.len()];
    for (i, &v) in ord.iter().enumerate() {
        res[v] = i;
    }
    res
}

fn gsea_stats1(
    stats: &[i64],
    selected_stats: &[usize], // 1-based
    selected_order: &[usize],
    gsea_param: f64,
    rev: bool,
) -> Vec<f64> {
    let n = stats.len();
    let k = selected_stats.len();
    let mut res = vec![0.0; k];

    let mut xs = SegmentTreeI64::new(k + 1);
    let mut ys = SegmentTreeF64::new(k + 1);

    let mut selected_ranks = ranks_from_order(selected_order);

    if !rev {
        let mut prev = -1i64;
        for i in 0..k {
            let j = selected_order[i];
            let t = selected_stats[j] as i64 - 1;
            xs.inc(i, t - prev);
            prev = t;
        }
        xs.inc(k, (n as i64) - 1 - prev);
    } else {
        let mut prev = n as i64;
        for i in 0..k {
            selected_ranks[i] = k - 1 - selected_ranks[i];
            let j = selected_order[k - 1 - i];
            let t = selected_stats[j] as i64 - 1;
            xs.inc(i, prev - t);
            prev = t;
        }
        xs.inc(k, prev);
    }

    let mut st_prev = vec![0isize; k + 2];
    let mut st_next = vec![0isize; k + 2];

    let k1 = ((k + 1) as f64).sqrt() as usize;
    let k2 = (k + 1) / k1 + 1;
    let mut block_summit = vec![0usize; k2];
    let mut block_start = vec![0usize; k2];
    let mut block_end = vec![0usize; k2];

    let mut i = 0usize;
    while i <= k + 1 {
        let block = i / k1;
        block_start[block] = i;
        block_end[block] = (i + k1 - 1).min(k + 1);
        for j in 1..=block_end[block] - i {
            st_prev[i + j] = (i + j - 1) as isize;
            st_next[i + j - 1] = (i + j) as isize;
        }
        st_prev[i] = i as isize;
        st_next[block_end[block]] = block_end[block] as isize;
        block_summit[block] = block_end[block];
        i += k1;
    }

    let mut stat_eps = 1e-5_f64;
    for &idx in selected_stats {
        let t = idx - 1;
        let xx = (stats[t].abs()) as f64;
        if xx > 0.0 {
            stat_eps = stat_eps.min(xx);
        }
    }
    stat_eps /= 1024.0;

    let unit_param = (gsea_param - 1.0).abs() < 1e-15;
    let mut nr = 0.0;
    for i in 0..k {
        let t = selected_stats[i] - 1;
        let t_rank = selected_ranks[i];
        let adj_stat = if unit_param {
            (stats[t].abs() as f64).max(stat_eps)
        } else {
            ((stats[t].abs() as f64).max(stat_eps)).powf(gsea_param)
        };

        xs.inc(t_rank, -1);
        ys.inc(t_rank, adj_stat);
        nr += adj_stat;

        let m = i + 1;
        let cur_block = (t_rank + 1) / k1;
        let b_s = block_start[cur_block];
        let b_e = block_end[cur_block];
        let mut cur_top = t_rank.max(b_s);

        for c in (t_rank + 1)..=b_e {
            let xc = xs.query_r(c) as f64;
            let yc = ys.query_r(c);

            let mut b = cur_top;
            let mut xb = xs.query_r(b) as f64;
            let mut yb = ys.query_r(b);

            while st_prev[cur_top] != cur_top as isize {
                let a = st_prev[cur_top] as usize;
                let xa = xs.query_r(a) as f64;
                let ya = ys.query_r(a);
                let mut pr = (xb - xa) * (yc - yb) - (yb - ya) * (xc - xb);
                if yc - ya < EPS {
                    pr = 0.0;
                }
                if pr <= 0.0 {
                    break;
                }
                cur_top = a;
                st_next[b] = -1;
                b = a;
                xb = xa;
                yb = ya;
            }

            st_prev[c] = cur_top as isize;
            st_next[cur_top] = c as isize;
            cur_top = c;
            if st_next[c] != -1 {
                break;
            }
        }

        let coef = (n - m) as f64 / nr;
        block_summit[cur_block] = block_summit[cur_block].max(cur_top);
        let mut max_p: f64 = 0.0;

        for block in 0..k2 {
            let mut cur_summit = block_summit[block];
            let mut cur_ys = ys.query_r(cur_summit);
            let mut cur_xs = xs.query_r(cur_summit) as f64;
            let mut cur_dist = cur_ys * coef - cur_xs;

            loop {
                let next_summit = st_prev[cur_summit] as usize;
                let next_ys = ys.query_r(next_summit);
                let next_xs = xs.query_r(next_summit) as f64;
                let next_dist = next_ys * coef - next_xs;
                if next_dist <= cur_dist {
                    break;
                }
                cur_dist = next_dist;
                cur_ys = next_ys;
                cur_xs = next_xs;
                cur_summit = next_summit;
            }

            block_summit[block] = cur_summit;
            max_p = max_p.max(cur_dist);
        }

        max_p /= (n - m) as f64;
        res[i] = max_p;
    }

    res
}

fn gsea_stats1_f64(
    stats: &[f64],
    selected_stats: &[usize], // 1-based
    selected_order: &[usize],
    gsea_param: f64,
    rev: bool,
) -> Vec<f64> {
    let n = stats.len();
    let k = selected_stats.len();
    let mut res = vec![0.0; k];

    let mut xs = SegmentTreeI64::new(k + 1);
    let mut ys = SegmentTreeF64::new(k + 1);

    let mut selected_ranks = ranks_from_order(selected_order);

    if !rev {
        let mut prev = -1i64;
        for i in 0..k {
            let j = selected_order[i];
            let t = selected_stats[j] as i64 - 1;
            xs.inc(i, t - prev);
            prev = t;
        }
        xs.inc(k, (n as i64) - 1 - prev);
    } else {
        let mut prev = n as i64;
        for i in 0..k {
            selected_ranks[i] = k - 1 - selected_ranks[i];
            let j = selected_order[k - 1 - i];
            let t = selected_stats[j] as i64 - 1;
            xs.inc(i, prev - t);
            prev = t;
        }
        xs.inc(k, prev);
    }

    let mut st_prev = vec![0isize; k + 2];
    let mut st_next = vec![0isize; k + 2];

    let k1 = ((k + 1) as f64).sqrt() as usize;
    let k2 = (k + 1) / k1 + 1;
    let mut block_summit = vec![0usize; k2];
    let mut block_start = vec![0usize; k2];
    let mut block_end = vec![0usize; k2];

    let mut i = 0usize;
    while i <= k + 1 {
        let block = i / k1;
        block_start[block] = i;
        block_end[block] = (i + k1 - 1).min(k + 1);
        for j in 1..=block_end[block] - i {
            st_prev[i + j] = (i + j - 1) as isize;
            st_next[i + j - 1] = (i + j) as isize;
        }
        st_prev[i] = i as isize;
        st_next[block_end[block]] = block_end[block] as isize;
        block_summit[block] = block_end[block];
        i += k1;
    }

    let mut stat_eps = 1e-5_f64;
    for &idx in selected_stats {
        let t = idx - 1;
        let xx = stats[t].abs();
        if xx > 0.0 {
            stat_eps = stat_eps.min(xx);
        }
    }
    stat_eps /= 1024.0;

    let unit_param = (gsea_param - 1.0).abs() < 1e-15;
    let mut nr = 0.0;
    for i in 0..k {
        let t = selected_stats[i] - 1;
        let t_rank = selected_ranks[i];
        let adj_stat = if unit_param {
            stats[t].abs().max(stat_eps)
        } else {
            (stats[t].abs().max(stat_eps)).powf(gsea_param)
        };

        xs.inc(t_rank, -1);
        ys.inc(t_rank, adj_stat);
        nr += adj_stat;

        let m = i + 1;
        let cur_block = (t_rank + 1) / k1;
        let b_s = block_start[cur_block];
        let b_e = block_end[cur_block];
        let mut cur_top = t_rank.max(b_s);

        for c in (t_rank + 1)..=b_e {
            let xc = xs.query_r(c) as f64;
            let yc = ys.query_r(c);

            let mut b = cur_top;
            let mut xb = xs.query_r(b) as f64;
            let mut yb = ys.query_r(b);

            while st_prev[cur_top] != cur_top as isize {
                let a = st_prev[cur_top] as usize;
                let xa = xs.query_r(a) as f64;
                let ya = ys.query_r(a);
                let mut pr = (xb - xa) * (yc - yb) - (yb - ya) * (xc - xb);
                if yc - ya < EPS {
                    pr = 0.0;
                }
                if pr <= 0.0 {
                    break;
                }
                cur_top = a;
                st_next[b] = -1;
                b = a;
                xb = xa;
                yb = ya;
            }

            st_prev[c] = cur_top as isize;
            st_next[cur_top] = c as isize;
            cur_top = c;
            if st_next[c] != -1 {
                break;
            }
        }

        let coef = (n - m) as f64 / nr;
        block_summit[cur_block] = block_summit[cur_block].max(cur_top);
        let mut max_p: f64 = 0.0;

        for block in 0..k2 {
            let mut cur_summit = block_summit[block];
            let mut cur_ys = ys.query_r(cur_summit);
            let mut cur_xs = xs.query_r(cur_summit) as f64;
            let mut cur_dist = cur_ys * coef - cur_xs;

            loop {
                let next_summit = st_prev[cur_summit] as usize;
                let next_ys = ys.query_r(next_summit);
                let next_xs = xs.query_r(next_summit) as f64;
                let next_dist = next_ys * coef - next_xs;
                if next_dist <= cur_dist {
                    break;
                }
                cur_dist = next_dist;
                cur_ys = next_ys;
                cur_xs = next_xs;
                cur_summit = next_summit;
            }

            block_summit[block] = cur_summit;
            max_p = max_p.max(cur_dist);
        }

        max_p /= (n - m) as f64;
        res[i] = max_p;
    }

    res
}

pub fn calc_gsea_stat_cumulative(
    stats: &[i64],
    selected_stats: &[usize], // 1-based
    gsea_param: f64,
    score_type: ScoreType,
) -> Vec<f64> {
    let selected_order = order(selected_stats);
    match score_type {
        ScoreType::Std => {
            let mut res = gsea_stats1(stats, selected_stats, &selected_order, gsea_param, false);
            let res_down = gsea_stats1(stats, selected_stats, &selected_order, gsea_param, true);
            for i in 0..selected_stats.len() {
                if res[i] == res_down[i] {
                    res[i] = 0.0;
                } else if res[i] < res_down[i] {
                    res[i] = -res_down[i];
                }
            }
            res
        }
        ScoreType::Pos => gsea_stats1(stats, selected_stats, &selected_order, gsea_param, false),
        ScoreType::Neg => gsea_stats1(stats, selected_stats, &selected_order, gsea_param, true)
            .into_iter()
            .map(|v| -v)
            .collect(),
    }
}

pub fn calc_gsea_stat_cumulative_f64(
    stats: &[f64],
    selected_stats: &[usize], // 1-based
    gsea_param: f64,
    score_type: ScoreType,
) -> Vec<f64> {
    let selected_order = order(selected_stats);
    match score_type {
        ScoreType::Std => {
            let mut res =
                gsea_stats1_f64(stats, selected_stats, &selected_order, gsea_param, false);
            let res_down =
                gsea_stats1_f64(stats, selected_stats, &selected_order, gsea_param, true);
            for i in 0..selected_stats.len() {
                if res[i] == res_down[i] {
                    res[i] = 0.0;
                } else if res[i] < res_down[i] {
                    res[i] = -res_down[i];
                }
            }
            res
        }
        ScoreType::Pos => {
            gsea_stats1_f64(stats, selected_stats, &selected_order, gsea_param, false)
        }
        ScoreType::Neg => gsea_stats1_f64(stats, selected_stats, &selected_order, gsea_param, true)
            .into_iter()
            .map(|v| -v)
            .collect(),
    }
}

pub struct BatchCounts {
    pub le_es: Vec<usize>,
    pub ge_es: Vec<usize>,
    pub le_zero: Vec<usize>,
    pub ge_zero: Vec<usize>,
    pub le_zero_sum: Vec<f64>,
    pub ge_zero_sum: Vec<f64>,
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn merge_batch_counts(mut acc: BatchCounts, rhs: BatchCounts) -> BatchCounts {
    for i in 0..acc.le_es.len() {
        acc.le_es[i] += rhs.le_es[i];
        acc.ge_es[i] += rhs.ge_es[i];
        acc.le_zero[i] += rhs.le_zero[i];
        acc.ge_zero[i] += rhs.ge_zero[i];
        acc.le_zero_sum[i] += rhs.le_zero_sum[i];
        acc.ge_zero_sum[i] += rhs.ge_zero_sum[i];
    }
    acc
}

#[inline]
fn empty_batch_counts(m: usize) -> BatchCounts {
    BatchCounts {
        le_es: vec![0; m],
        ge_es: vec![0; m],
        le_zero: vec![0; m],
        ge_zero: vec![0; m],
        le_zero_sum: vec![0.0; m],
        ge_zero_sum: vec![0.0; m],
    }
}

#[inline]
fn chunk_ranges(iterations: usize) -> Vec<(usize, Range<usize>)> {
    if iterations == 0 {
        return Vec::new();
    }
    let workers = rayon::current_num_threads().max(1);
    let chunks = iterations.min((workers * 8).max(1));
    let base = iterations / chunks;
    let rem = iterations % chunks;
    let mut ranges = Vec::with_capacity(chunks);
    let mut start = 0usize;
    for ci in 0..chunks {
        let len = base + usize::from(ci < rem);
        let end = start + len;
        ranges.push((ci, Range { start, end }));
        start = end;
    }
    ranges
}

#[inline]
fn sample_k_small_reuse(
    n: usize,
    k: usize,
    rng: &mut Mt19937Compat,
    marks: &mut [u32],
    stamp: &mut u32,
    out: &mut Vec<usize>,
) {
    *stamp = stamp.wrapping_add(1);
    if *stamp == 0 {
        marks.fill(0);
        *stamp = 1;
    }
    out.clear();
    while out.len() < k {
        let x = uid_wrapper(1, n, rng);
        let idx = x - 1;
        if marks[idx] != *stamp {
            marks[idx] = *stamp;
            out.push(x);
        }
    }
}

pub fn calc_gsea_stat_cumulative_batch(
    stats: &[i64],
    gsea_param: f64,
    pathway_scores: &[f64],
    pathways_sizes: &[usize], // 1-based indices into cumulative vector
    iterations: usize,
    seed: u64,
    score_type: ScoreType,
) -> BatchCounts {
    let n = stats.len();
    let k = *pathways_sizes.iter().max().unwrap_or(&0);
    let m = pathways_sizes.len();
    let mut out = empty_batch_counts(m);
    if iterations == 0 || m == 0 || k == 0 {
        return out;
    }
    let mut rng = Mt19937Compat::new(seed as u32);
    for _ in 0..iterations {
        let selected = combination(1, n, k, &mut rng); // 1-based
        let rand_es = calc_gsea_stat_cumulative(stats, &selected, gsea_param, score_type);
        for i in 0..m {
            let v = rand_es[pathways_sizes[i] - 1];
            if v <= pathway_scores[i] {
                out.le_es[i] += 1;
            }
            if v >= pathway_scores[i] {
                out.ge_es[i] += 1;
            }
            if v <= 0.0 {
                out.le_zero[i] += 1;
                out.le_zero_sum[i] += v;
            }
            if v >= 0.0 {
                out.ge_zero[i] += 1;
                out.ge_zero_sum[i] += v;
            }
        }
    }
    out
}

pub fn calc_gsea_stat_cumulative_batch_parallel(
    stats: &[i64],
    gsea_param: f64,
    pathway_scores: &[f64],
    pathways_sizes: &[usize], // 1-based indices into cumulative vector
    iterations: usize,
    seed: u64,
    score_type: ScoreType,
) -> BatchCounts {
    let n = stats.len();
    let k = *pathways_sizes.iter().max().unwrap_or(&0);
    let m = pathways_sizes.len();
    let empty = empty_batch_counts(m);
    if iterations == 0 || m == 0 || k == 0 {
        return empty;
    }
    let score_indices: Vec<usize> = pathways_sizes.iter().map(|&x| x - 1).collect();
    let pool_template: Vec<usize> = (1..=n).collect();

    chunk_ranges(iterations)
        .into_par_iter()
        .map(|(chunk_idx, iteration_range)| {
            let mut out = empty_batch_counts(m);
            let chunk_seed =
                splitmix64(seed ^ ((chunk_idx as u64) << 1) ^ (iteration_range.start as u64));
            let mut rng = Mt19937Compat::new(chunk_seed as u32);
            let mut pool = pool_template.clone();
            let mut marks = vec![0u32; n];
            let mut stamp = 0u32;
            let mut selected = vec![0usize; k];
            for _ in iteration_range {
                if k > n / 2 {
                    pool.copy_from_slice(&pool_template);
                    for i in (0..k).rev() {
                        let r = n - 1 - i;
                        let x = uid_wrapper(0, r, &mut rng);
                        let j = i + x;
                        pool.swap(i, j);
                    }
                    selected.copy_from_slice(&pool[..k]);
                } else {
                    sample_k_small_reuse(n, k, &mut rng, &mut marks, &mut stamp, &mut selected);
                }

                let rand_es = calc_gsea_stat_cumulative(stats, &selected, gsea_param, score_type);
                for i in 0..m {
                    let v = rand_es[score_indices[i]];
                    if v <= pathway_scores[i] {
                        out.le_es[i] += 1;
                    }
                    if v >= pathway_scores[i] {
                        out.ge_es[i] += 1;
                    }
                    if v <= 0.0 {
                        out.le_zero[i] += 1;
                        out.le_zero_sum[i] += v;
                    }
                    if v >= 0.0 {
                        out.ge_zero[i] += 1;
                        out.ge_zero_sum[i] += v;
                    }
                }
            }
            out
        })
        .reduce(|| empty_batch_counts(m), merge_batch_counts)
}

pub fn calc_gsea_stat_cumulative_batch_f64(
    stats: &[f64],
    gsea_param: f64,
    pathway_scores: &[f64],
    pathways_sizes: &[usize], // 1-based indices into cumulative vector
    iterations: usize,
    seed: u64,
    score_type: ScoreType,
) -> BatchCounts {
    let n = stats.len();
    let k = *pathways_sizes.iter().max().unwrap_or(&0);
    let m = pathways_sizes.len();
    let mut out = empty_batch_counts(m);
    if iterations == 0 || m == 0 || k == 0 {
        return out;
    }
    let mut rng = Mt19937Compat::new(seed as u32);
    for _ in 0..iterations {
        let selected = combination(1, n, k, &mut rng); // 1-based
        let rand_es = calc_gsea_stat_cumulative_f64(stats, &selected, gsea_param, score_type);
        for i in 0..m {
            let v = rand_es[pathways_sizes[i] - 1];
            if v <= pathway_scores[i] {
                out.le_es[i] += 1;
            }
            if v >= pathway_scores[i] {
                out.ge_es[i] += 1;
            }
            if v <= 0.0 {
                out.le_zero[i] += 1;
                out.le_zero_sum[i] += v;
            }
            if v >= 0.0 {
                out.ge_zero[i] += 1;
                out.ge_zero_sum[i] += v;
            }
        }
    }
    out
}

#[derive(Clone, Copy, Default)]
struct PathwayAcc {
    le_es: usize,
    ge_es: usize,
    le_zero: usize,
    ge_zero: usize,
    le_zero_sum: f64,
    ge_zero_sum: f64,
}

struct SizeGroup {
    score_index: usize,
    pathway_indices: Vec<usize>,
}

fn grouped_score_indices(pathways_sizes: &[usize]) -> Vec<SizeGroup> {
    let mut by_score_index: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (pathway_idx, &pathway_size) in pathways_sizes.iter().enumerate() {
        by_score_index
            .entry(pathway_size - 1)
            .or_default()
            .push(pathway_idx);
    }

    by_score_index
        .into_iter()
        .map(|(score_index, pathway_indices)| SizeGroup {
            score_index,
            pathway_indices,
        })
        .collect()
}

pub fn calc_gsea_stat_cumulative_batch_f64_thread_invariant_parallel(
    stats: &[f64],
    gsea_param: f64,
    pathway_scores: &[f64],
    pathways_sizes: &[usize], // 1-based indices into cumulative vector
    iterations: usize,
    seed: u64,
    score_type: ScoreType,
) -> BatchCounts {
    let n = stats.len();
    let k = *pathways_sizes.iter().max().unwrap_or(&0);
    let m = pathways_sizes.len();
    if iterations == 0 || m == 0 || k == 0 {
        return empty_batch_counts(m);
    }

    let size_groups = grouped_score_indices(pathways_sizes);
    let mut group_accs: Vec<Vec<PathwayAcc>> = size_groups
        .iter()
        .map(|group| vec![PathwayAcc::default(); group.pathway_indices.len()])
        .collect();
    let mut rng = Mt19937Compat::new(seed as u32);
    const BLOCK_ITERS: usize = 4096;
    let mut done = 0usize;
    let mut rand_es_block: Vec<Vec<f64>> = Vec::with_capacity(BLOCK_ITERS);

    // Thread-invariant strategy:
    // - keep RNG and iteration order exactly serial
    // - parallelize pathway updates over blocks of iterations
    // - group pathways by size so each cumulative ES vector is indexed once
    //   per unique size instead of once per pathway
    // This preserves single-core numerical behavior while using multiple cores.
    while done < iterations {
        rand_es_block.clear();
        let block_end = (done + BLOCK_ITERS).min(iterations);
        for _ in done..block_end {
            let selected = combination(1, n, k, &mut rng); // 1-based
            rand_es_block.push(calc_gsea_stat_cumulative_f64(
                stats, &selected, gsea_param, score_type,
            ));
        }

        group_accs
            .par_iter_mut()
            .zip(size_groups.par_iter())
            .for_each(|(accs, group)| {
                let pathway_idxs = &group.pathway_indices;
                for rand_es in &rand_es_block {
                    let v = rand_es[group.score_index];
                    let v_le_zero = v <= 0.0;
                    let v_ge_zero = v >= 0.0;
                    for (local_i, acc) in accs.iter_mut().enumerate() {
                        let pathway_score = pathway_scores[pathway_idxs[local_i]];
                        if v <= pathway_score {
                            acc.le_es += 1;
                        }
                        if v >= pathway_score {
                            acc.ge_es += 1;
                        }
                        if v_le_zero {
                            acc.le_zero += 1;
                            acc.le_zero_sum += v;
                        }
                        if v_ge_zero {
                            acc.ge_zero += 1;
                            acc.ge_zero_sum += v;
                        }
                    }
                }
            });
        done = block_end;
    }

    let mut out = empty_batch_counts(m);
    for (group, accs) in size_groups.iter().zip(group_accs) {
        for (&pathway_idx, acc) in group.pathway_indices.iter().zip(accs) {
            out.le_es[pathway_idx] = acc.le_es;
            out.ge_es[pathway_idx] = acc.ge_es;
            out.le_zero[pathway_idx] = acc.le_zero;
            out.ge_zero[pathway_idx] = acc.ge_zero;
            out.le_zero_sum[pathway_idx] = acc.le_zero_sum;
            out.ge_zero_sum[pathway_idx] = acc.ge_zero_sum;
        }
    }
    out
}

pub fn calc_gsea_stat_cumulative_batch_f64_parallel(
    stats: &[f64],
    gsea_param: f64,
    pathway_scores: &[f64],
    pathways_sizes: &[usize], // 1-based indices into cumulative vector
    iterations: usize,
    seed: u64,
    score_type: ScoreType,
) -> BatchCounts {
    let n = stats.len();
    let k = *pathways_sizes.iter().max().unwrap_or(&0);
    let m = pathways_sizes.len();
    let empty = empty_batch_counts(m);
    if iterations == 0 || m == 0 || k == 0 {
        return empty;
    }
    let score_indices: Vec<usize> = pathways_sizes.iter().map(|&x| x - 1).collect();
    let pool_template: Vec<usize> = (1..=n).collect();

    chunk_ranges(iterations)
        .into_par_iter()
        .map(|(chunk_idx, iteration_range)| {
            let mut out = empty_batch_counts(m);
            let chunk_seed =
                splitmix64(seed ^ ((chunk_idx as u64) << 1) ^ (iteration_range.start as u64));
            let mut rng = Mt19937Compat::new(chunk_seed as u32);
            let mut pool = pool_template.clone();
            let mut marks = vec![0u32; n];
            let mut stamp = 0u32;
            let mut selected = vec![0usize; k];
            for _ in iteration_range {
                if k > n / 2 {
                    pool.copy_from_slice(&pool_template);
                    for i in (0..k).rev() {
                        let r = n - 1 - i;
                        let x = uid_wrapper(0, r, &mut rng);
                        let j = i + x;
                        pool.swap(i, j);
                    }
                    selected.copy_from_slice(&pool[..k]);
                } else {
                    sample_k_small_reuse(n, k, &mut rng, &mut marks, &mut stamp, &mut selected);
                }

                let rand_es =
                    calc_gsea_stat_cumulative_f64(stats, &selected, gsea_param, score_type);
                for i in 0..m {
                    let v = rand_es[score_indices[i]];
                    if v <= pathway_scores[i] {
                        out.le_es[i] += 1;
                    }
                    if v >= pathway_scores[i] {
                        out.ge_es[i] += 1;
                    }
                    if v <= 0.0 {
                        out.le_zero[i] += 1;
                        out.le_zero_sum[i] += v;
                    }
                    if v >= 0.0 {
                        out.ge_zero[i] += 1;
                        out.ge_zero_sum[i] += v;
                    }
                }
            }
            out
        })
        .reduce(|| empty_batch_counts(m), merge_batch_counts)
}
