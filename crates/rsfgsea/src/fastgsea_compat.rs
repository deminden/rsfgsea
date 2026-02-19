#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use crate::core::ScoreType;
use crate::rng_compat::{Mt19937Compat, combination};

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

    fn query_r(&self, mut r: usize) -> i64 {
        if r == 0 {
            return 0;
        }
        r -= 1;
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

    fn query_r(&self, mut r: usize) -> f64 {
        if r == 0 {
            return 0.0;
        }
        r -= 1;
        self.t[r] + self.b[r >> self.log_k]
    }
}

fn order(x: &[usize]) -> Vec<usize> {
    let mut res: Vec<usize> = (0..x.len()).collect();
    res.sort_by_key(|&i| x[i]);
    res
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

    let mut nr = 0.0;
    for i in 0..k {
        let t = selected_stats[i] - 1;
        let t_rank = selected_ranks[i];
        let adj_stat = ((stats[t].abs() as f64).max(stat_eps)).powf(gsea_param);

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
            let mut cur_dist = ys.query_r(cur_summit) * coef - (xs.query_r(cur_summit) as f64);

            loop {
                let next_summit = st_prev[cur_summit] as usize;
                let next_dist = ys.query_r(next_summit) * coef - (xs.query_r(next_summit) as f64);
                if next_dist <= cur_dist {
                    break;
                }
                cur_dist = next_dist;
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

    let mut nr = 0.0;
    for i in 0..k {
        let t = selected_stats[i] - 1;
        let t_rank = selected_ranks[i];
        let adj_stat = (stats[t].abs().max(stat_eps)).powf(gsea_param);

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
            let mut cur_dist = ys.query_r(cur_summit) * coef - (xs.query_r(cur_summit) as f64);

            loop {
                let next_summit = st_prev[cur_summit] as usize;
                let next_dist = ys.query_r(next_summit) * coef - (xs.query_r(next_summit) as f64);
                if next_dist <= cur_dist {
                    break;
                }
                cur_dist = next_dist;
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
    let mut out = BatchCounts {
        le_es: vec![0; m],
        ge_es: vec![0; m],
        le_zero: vec![0; m],
        ge_zero: vec![0; m],
        le_zero_sum: vec![0.0; m],
        ge_zero_sum: vec![0.0; m],
    };
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
    let mut out = BatchCounts {
        le_es: vec![0; m],
        ge_es: vec![0; m],
        le_zero: vec![0; m],
        ge_zero: vec![0; m],
        le_zero_sum: vec![0.0; m],
        ge_zero_sum: vec![0.0; m],
    };
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
