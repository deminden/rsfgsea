#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(clippy::needless_range_loop)]

use crate::rng_compat::{Mt19937Compat, combination, uid_wrapper};
use special::Gamma;

pub fn beta_mean_log(a: usize, b: usize) -> f64 {
    if a > b {
        return 0.0;
    }
    let mut s = 0.0_f64;
    for i in a..=b {
        s -= 1.0 / (i as f64);
    }
    s
}

pub fn multilevel_error_level(level: usize, sample_size: usize) -> f64 {
    let single_level_error =
        (((sample_size + 1) as f64) / 2.0).trigamma() - ((sample_size + 1) as f64).trigamma();
    ((level as f64) * single_level_error).sqrt() / 2.0_f64.ln()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Score {
    pub ns: i64,
    pub coef_ns: i64,
    pub diff: i64,
    pub coef_const: i64,
}

impl Score {
    pub fn get_double(self) -> f64 {
        (self.coef_ns as f64 / self.ns as f64) - (self.coef_const as f64 / self.diff as f64)
    }

    pub fn numerator(self) -> i64 {
        self.coef_ns * self.diff - self.coef_const * self.ns
    }

    pub fn compare_raw(self, other: Self) -> i128 {
        let p1 = self.coef_ns * self.diff + self.ns * (other.coef_const - self.coef_const);
        let q1 = self.ns * self.diff;
        let p2 = other.coef_ns;
        let q2 = other.ns;
        (p1 as i128) * (q2 as i128) - (p2 as i128) * (q1 as i128)
    }

    pub fn lt_cpp(self, other: Self) -> bool {
        self.compare_raw(other) < 0
    }

    pub fn le_cpp(self, other: Self) -> bool {
        self.compare_raw(other) <= 0
    }

    pub fn gt_cpp(self, other: Self) -> bool {
        self.compare_raw(other) > 0
    }

    pub fn ge_cpp(self, other: Self) -> bool {
        self.compare_raw(other) >= 0
    }

    pub fn max_ns() -> i64 {
        1_i64 << 30
    }

    pub fn abs(self) -> Self {
        if self >= -self { self } else { -self }
    }
}

impl std::ops::Neg for Score {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            ns: self.ns,
            coef_ns: -self.coef_ns,
            diff: self.diff,
            coef_const: -self.coef_const,
        }
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.compare_raw(*other).cmp(&0)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gsea {
    pub score: Score,
    pub hash: u64,
}

impl Gsea {
    // Match std::pair<score_t, hash_t> comparison semantics from C++:
    // a < b iff (a.score < b.score) || (!(b.score < a.score) && a.hash < b.hash)
    pub fn lt_cpp(self, other: Self) -> bool {
        if self.score.lt_cpp(other.score) {
            true
        } else if other.score.lt_cpp(self.score) {
            false
        } else {
            self.hash < other.hash
        }
    }

    pub fn ge_cpp(self, other: Self) -> bool {
        !self.lt_cpp(other)
    }

    pub fn le_cpp(self, other: Self) -> bool {
        !other.lt_cpp(self)
    }

    pub fn cmp_cpp(self, other: Self) -> std::cmp::Ordering {
        if self.lt_cpp(other) {
            std::cmp::Ordering::Less
        } else if other.lt_cpp(self) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

pub fn calc_es(ranks: &[i64], p: &[usize], ns_opt: Option<i64>) -> Score {
    let n = ranks.len() as i64;
    let k = p.len() as i64;
    let ns = ns_opt.unwrap_or_else(|| p.iter().map(|&pos| ranks[pos]).sum());

    let mut res = Score {
        ns,
        coef_ns: 0,
        diff: n - k,
        coef_const: 0,
    };
    let mut cur = res;
    let mut last = -1_i64;

    for &pos_u in p {
        let pos = pos_u as i64;
        cur.coef_const += pos - last - 1;
        if res.abs() < cur.abs() {
            res = cur;
        }
        cur.coef_ns += ranks[pos_u];
        if res.abs() < cur.abs() {
            res = cur;
        }
        last = pos;
    }

    res
}

pub fn calc_positive_es(ranks: &[i64], p: &[usize], ns_opt: Option<i64>) -> Score {
    let n = ranks.len() as i64;
    let k = p.len() as i64;
    let ns = ns_opt.unwrap_or_else(|| p.iter().map(|&pos| ranks[pos]).sum());

    let mut res = Score {
        ns,
        coef_ns: 0,
        diff: n - k,
        coef_const: 0,
    };
    let mut cur = res;
    let mut last = -1_i64;

    for &pos_u in p {
        let pos = pos_u as i64;
        cur.coef_ns += ranks[pos_u];
        cur.coef_const += pos - last - 1;
        if cur > res {
            res = cur;
        }
        last = pos;
    }

    res
}

#[derive(Clone, Debug)]
struct Level {
    low_scores: Vec<(Gsea, bool)>,
    high_scores: Vec<(Gsea, bool)>,
    bound: Gsea,
}

#[derive(Clone, Debug)]
struct SampleChunks {
    chunk_sum: Vec<i64>,
    chunks: Vec<Vec<usize>>,
}

impl SampleChunks {
    fn new(chunks_number: usize) -> Self {
        Self {
            chunk_sum: vec![0; chunks_number],
            chunks: vec![Vec::new(); chunks_number],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PerturbateResult {
    moves: i32,
    iters: i32,
}

#[derive(Clone, Debug)]
pub struct EsRulerCompat {
    log_status: bool,
    ranks: Vec<i64>,
    gene_hashes: Vec<u64>,
    sample_size: usize,
    pathway_size: usize,
    moves_scale: f64,
    incorrect_ruler: bool,
    current_samples: Vec<Vec<usize>>,
    old_samples_start: usize,
    levels: Vec<Level>,
    chunk_last_element: Vec<usize>,
    chunks_number: usize,
}

impl EsRulerCompat {
    pub fn new(
        ranks: Vec<i64>,
        sample_size: usize,
        pathway_size: usize,
        moves_scale: f64,
        log_status: bool,
    ) -> Self {
        Self {
            log_status,
            gene_hashes: vec![0; ranks.len()],
            ranks,
            sample_size,
            pathway_size,
            moves_scale,
            incorrect_ruler: false,
            current_samples: vec![Vec::new(); sample_size],
            old_samples_start: 0,
            levels: Vec::new(),
            chunk_last_element: Vec::new(),
            chunks_number: 0,
        }
    }

    fn calc_hash(&self, sample: &[usize]) -> u64 {
        let mut res = 0_u64;
        for &i in sample {
            res ^= self.gene_hashes[i];
        }
        res
    }

    fn resample_genesets(&mut self, rng: &mut Mt19937Compat) -> bool {
        let mut stats: Vec<(Gsea, bool, usize)> = Vec::with_capacity(self.sample_size);
        for sample_id in 0..self.sample_size {
            let sample_es_pos =
                calc_positive_es(&self.ranks, &self.current_samples[sample_id], None);
            let sample_hash = self.calc_hash(&self.current_samples[sample_id]);
            // Match fgsea C++ exactly: positivity flag comes from calcES().getNumerator() >= 0.
            let sample_es = calc_es(&self.ranks, &self.current_samples[sample_id], None);
            let sample_is_positive = sample_es.numerator() >= 0;
            stats.push((
                Gsea {
                    score: sample_es_pos,
                    hash: sample_hash,
                },
                sample_is_positive,
                sample_id,
            ));
        }
        stats.sort_by(|a, b| {
            a.0.cmp_cpp(b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let central_value = stats[self.sample_size / 2].0;
        let mut start_from = 0usize;
        for (i, s) in stats.iter().enumerate() {
            if s.0.ge_cpp(central_value) {
                start_from = i;
                break;
            }
        }

        if start_from == 0 {
            while start_from < self.sample_size
                && stats[start_from].0.cmp_cpp(stats[0].0) == std::cmp::Ordering::Equal
            {
                start_from += 1;
            }
        }

        if start_from == self.sample_size {
            return true;
        }

        let mut level = Level {
            low_scores: Vec::new(),
            high_scores: Vec::new(),
            bound: stats[start_from - 1].0,
        };

        for s in stats.iter().take(start_from) {
            level.low_scores.push((s.0, s.1));
        }
        for s in stats.iter().skip(start_from) {
            level.high_scores.push((s.0, s.1));
        }
        self.levels.push(level);

        let mut new_sets: Vec<Vec<usize>> = Vec::with_capacity(self.sample_size);
        for _ in 0..start_from {
            let ind = uid_wrapper(0, self.sample_size - start_from - 1, rng) + start_from;
            new_sets.push(self.current_samples[stats[ind].2].clone());
        }
        for s in stats.iter().skip(start_from) {
            new_sets.push(self.current_samples[s.2].clone());
        }

        self.old_samples_start = start_from;
        self.current_samples = new_sets;
        true
    }

    pub fn extend(&mut self, es_double: f64, seed: u64, eps: f64) {
        let mut rng = Mt19937Compat::new(seed as u32);
        let n = self.ranks.len();
        let k = self.pathway_size;

        for i in 0..n {
            self.gene_hashes[i] = rng.next_u32() as u64;
        }

        for sample_id in 0..self.sample_size {
            let mut s = combination(0, self.ranks.len() - 1, self.pathway_size, &mut rng);
            s.sort_unstable();
            self.current_samples[sample_id] = s;
        }

        if !self.resample_genesets(&mut rng) {
            self.incorrect_ruler = true;
            return;
        }

        self.chunks_number = ((self.pathway_size as f64).sqrt() as usize).max(1);
        self.chunk_last_element = vec![0usize; self.chunks_number];
        self.chunk_last_element[self.chunks_number - 1] = self.ranks.len();

        let need_es = Score {
            ns: Score::max_ns(),
            coef_ns: (Score::max_ns() as f64 * es_double) as i64,
            diff: 1,
            coef_const: 0,
        };

        let mut adj_log_pval = 0.0;
        let mut level_num = 1usize;

        while self
            .levels
            .last()
            .map(|l| l.bound.score < need_es)
            .unwrap_or(false)
        {
            let high_len = self.levels.last().unwrap().high_scores.len() + 1;
            adj_log_pval += beta_mean_log(high_len, self.sample_size);
            if eps != 0.0 && adj_log_pval < eps.ln() {
                break;
            }

            for i in 0..(self.chunks_number - 1) {
                let pos = (0..=i)
                    .map(|j| (self.pathway_size + j) / self.chunks_number)
                    .sum::<usize>();
                let mut tmp: Vec<usize> = (0..self.sample_size)
                    .map(|j| self.current_samples[j][pos])
                    .collect();
                tmp.select_nth_unstable(self.sample_size / 2);
                self.chunk_last_element[i] = tmp[self.sample_size / 2];
            }

            let mut samples_chunks: Vec<SampleChunks> = (0..self.sample_size)
                .map(|_| SampleChunks::new(self.chunks_number))
                .collect();

            for i in 0..self.sample_size {
                let mut cnt = 0usize;
                for &pos in &self.current_samples[i] {
                    while cnt < self.chunk_last_element.len() && self.chunk_last_element[cnt] <= pos
                    {
                        cnt += 1;
                    }
                    let c = cnt.min(self.chunk_last_element.len() - 1);
                    samples_chunks[i].chunks[c].push(pos);
                    samples_chunks[i].chunk_sum[c] += self.ranks[pos];
                }
            }

            let mut n_iterations = 0;
            let mut n_accepted = 0;
            let need_accepted =
                (self.moves_scale * self.sample_size as f64 * self.pathway_size as f64 / 2.0)
                    as i32;

            while n_accepted < need_accepted {
                for chunk in samples_chunks.iter_mut().take(self.sample_size) {
                    let pr = self.perturbate(k, chunk, self.levels.last().unwrap().bound, &mut rng);
                    n_accepted += pr.moves;
                }
                n_iterations += 1;
            }

            for _ in 0..n_iterations {
                for chunk in samples_chunks.iter_mut().take(self.sample_size) {
                    let _ = self.perturbate(k, chunk, self.levels.last().unwrap().bound, &mut rng);
                }
            }

            for (i, chunk) in samples_chunks.iter().enumerate().take(self.sample_size) {
                self.current_samples[i].clear();
                for c in &chunk.chunks {
                    self.current_samples[i].extend_from_slice(c);
                }
            }

            let last_size = self.levels.len();
            if !self.resample_genesets(&mut rng) {
                self.incorrect_ruler = true;
            }
            if last_size == self.levels.len() {
                break;
            }

            level_num += 1;
            let _ = level_num;
        }
    }

    pub fn get_pvalue(&self, es_double: f64, _eps: f64, sign: bool) -> (f64, bool, f64) {
        if self.incorrect_ruler || self.levels.is_empty() {
            return (f64::NAN, true, f64::NAN);
        }

        let es_score = Score {
            ns: Score::max_ns(),
            coef_ns: (Score::max_ns() as f64 * es_double) as i64,
            diff: 1,
            coef_const: 0,
        };
        let es = Gsea {
            score: es_score,
            hash: 0,
        };

        let mut adj_log_pval = 0.0;
        let mut lvls_var = 0.0;

        for lvl in &self.levels {
            if es.le_cpp(lvl.bound) {
                let mut cnt_last = 0usize;
                let mut cnt_positive = 0usize;
                for &(_, is_positive) in &lvl.high_scores {
                    cnt_last += 1;
                    cnt_positive += is_positive as usize;
                }
                for &(x, is_positive) in &lvl.low_scores {
                    if x.ge_cpp(es) {
                        cnt_last += 1;
                        cnt_positive += is_positive as usize;
                    }
                }

                let numerator = if sign { cnt_last } else { cnt_positive };
                if numerator == 0 {
                    adj_log_pval += beta_mean_log(1, self.sample_size);
                    return (adj_log_pval.exp().clamp(0.0, 1.0), true, f64::NAN);
                }

                adj_log_pval += beta_mean_log(numerator, self.sample_size);
                lvls_var +=
                    (numerator as f64).trigamma() - ((self.sample_size + 1) as f64).trigamma();

                return (
                    adj_log_pval.exp().clamp(0.0, 1.0),
                    true,
                    lvls_var.sqrt() / 2.0_f64.ln(),
                );
            }

            let nhigh = lvl.high_scores.len() + 1;
            adj_log_pval += beta_mean_log(nhigh, self.sample_size);
            lvls_var += (nhigh as f64).trigamma() - ((self.sample_size + 1) as f64).trigamma();
        }

        let last = self.levels.last().unwrap();
        let mut cnt_last = 0usize;
        let mut cnt_positive = 0usize;
        for &(x, is_positive) in &last.high_scores {
            if x.ge_cpp(es) {
                cnt_last += 1;
                cnt_positive += is_positive as usize;
            }
        }

        let numerator = if sign { cnt_last } else { cnt_positive };
        if numerator == 0 {
            adj_log_pval += beta_mean_log(1, last.high_scores.len());
            return (adj_log_pval.exp().clamp(0.0, 1.0), true, f64::NAN);
        }

        adj_log_pval += beta_mean_log(numerator, last.high_scores.len());
        lvls_var +=
            (numerator as f64).trigamma() - ((last.high_scores.len() + 1) as f64).trigamma();

        (
            adj_log_pval.exp().clamp(0.0, 1.0),
            true,
            lvls_var.sqrt() / 2.0_f64.ln(),
        )
    }

    fn perturbate(
        &self,
        k: usize,
        sample_chunks: &mut SampleChunks,
        bound: Gsea,
        rng: &mut Mt19937Compat,
    ) -> PerturbateResult {
        let iters = (k as f64 * 0.1) as i32;
        self.perturbate_iters(k, sample_chunks, bound, rng, iters.max(1))
    }

    fn perturbate_iters(
        &self,
        k: usize,
        sample_chunks: &mut SampleChunks,
        bound: Gsea,
        rng: &mut Mt19937Compat,
        need_iters: i32,
    ) -> PerturbateResult {
        self.perturbate_until(k, sample_chunks, bound, rng, |_, iters| iters >= need_iters)
    }

    #[allow(dead_code)]
    fn perturbate_success(
        &self,
        k: usize,
        sample_chunks: &mut SampleChunks,
        bound: Gsea,
        rng: &mut Mt19937Compat,
        need_successes: i32,
    ) -> PerturbateResult {
        self.perturbate_until(k, sample_chunks, bound, rng, |moves, _| {
            moves >= need_successes
        })
    }

    fn perturbate_until<F: Fn(i32, i32) -> bool>(
        &self,
        k: usize,
        sample_chunks: &mut SampleChunks,
        bound: Gsea,
        rng: &mut Mt19937Compat,
        stop: F,
    ) -> PerturbateResult {
        let n = self.ranks.len();

        let mut ns = 0_i64;
        let mut cur_hash = 0_u64;
        for ch in &sample_chunks.chunks {
            for &pos in ch {
                ns += self.ranks[pos];
                cur_hash ^= self.gene_hashes[pos];
            }
        }

        let mut cand_val: isize = -1;
        let mut has_cand = false;
        let mut cand_x = 0_i64;
        let mut cand_y = 0_i64;

        let mut moves = 0_i32;
        let mut iters = 0_i32;

        while !stop(moves, iters) {
            iters += 1;
            let old_ind = uid_wrapper(0, k - 1, rng);

            let mut old_chunk_ind = 0usize;
            let mut old_ind_in_chunk = 0usize;
            let old_val: usize;
            {
                let mut tmp = old_ind;
                while sample_chunks.chunks[old_chunk_ind].len() <= tmp {
                    tmp -= sample_chunks.chunks[old_chunk_ind].len();
                    old_chunk_ind += 1;
                }
                old_ind_in_chunk = tmp;
                old_val = sample_chunks.chunks[old_chunk_ind][old_ind_in_chunk];
            }

            let new_val = uid_wrapper(0, n - 1, rng);
            let new_chunk_ind = self.chunk_last_element.partition_point(|&x| x <= new_val);
            let new_chunk = new_chunk_ind.min(sample_chunks.chunks.len() - 1);
            let insert_pos = match sample_chunks.chunks[new_chunk].binary_search(&new_val) {
                Ok(pos) => pos,
                Err(pos) => pos,
            };

            if insert_pos < sample_chunks.chunks[new_chunk].len()
                && sample_chunks.chunks[new_chunk][insert_pos] == new_val
            {
                if new_val == old_val {
                    moves += 1;
                }
                continue;
            }

            sample_chunks.chunks[old_chunk_ind].remove(old_ind_in_chunk);
            let adj_insert = if old_chunk_ind == new_chunk && old_ind_in_chunk < insert_pos {
                insert_pos - 1
            } else {
                insert_pos
            };
            sample_chunks.chunks[new_chunk].insert(adj_insert, new_val);

            ns = ns - self.ranks[old_val] + self.ranks[new_val];
            cur_hash ^= self.gene_hashes[old_val] ^ self.gene_hashes[new_val];
            sample_chunks.chunk_sum[old_chunk_ind] -= self.ranks[old_val];
            sample_chunks.chunk_sum[new_chunk] += self.ranks[new_val];

            let strictly = cur_hash <= bound.hash;
            let check = |score: Score| {
                if strictly {
                    score > bound.score
                } else {
                    score >= bound.score
                }
            };

            if has_cand && old_val as isize == cand_val {
                has_cand = false;
            }
            if has_cand {
                if (old_val as i64) < cand_val as i64 {
                    cand_x += 1;
                    cand_y -= self.ranks[old_val];
                }
                if (new_val as i64) < cand_val as i64 {
                    cand_x -= 1;
                    cand_y += self.ranks[new_val];
                }
            }

            if has_cand
                && check(Score {
                    ns,
                    coef_ns: cand_y,
                    diff: (n - k) as i64,
                    coef_const: cand_x,
                })
            {
                moves += 1;
                continue;
            }

            let mut cur_x = 0_i64;
            let mut cur_y = 0_i64;
            let mut ok = false;
            let mut last = -1_i64;

            for i in 0..sample_chunks.chunks.len() {
                if !check(Score {
                    ns,
                    coef_ns: cur_y + sample_chunks.chunk_sum[i],
                    diff: (n - k) as i64,
                    coef_const: cur_x,
                }) {
                    cur_y += sample_chunks.chunk_sum[i];
                    cur_x += (self.chunk_last_element[i] as i64)
                        - last
                        - 1
                        - (sample_chunks.chunks[i].len() as i64);
                    last = self.chunk_last_element[i] as i64 - 1;
                } else {
                    for &pos in &sample_chunks.chunks[i] {
                        cur_y += self.ranks[pos];
                        cur_x += pos as i64 - last - 1;
                        if check(Score {
                            ns,
                            coef_ns: cur_y,
                            diff: (n - k) as i64,
                            coef_const: cur_x,
                        }) {
                            ok = true;
                            has_cand = true;
                            cand_x = cur_x;
                            cand_y = cur_y;
                            cand_val = pos as isize;
                            break;
                        }
                        last = pos as i64;
                    }
                    if ok {
                        break;
                    }
                    cur_x += self.chunk_last_element[i] as i64 - 1 - last;
                    last = self.chunk_last_element[i] as i64 - 1;
                }
            }

            if !ok {
                ns = ns - self.ranks[new_val] + self.ranks[old_val];
                cur_hash ^= self.gene_hashes[new_val] ^ self.gene_hashes[old_val];

                sample_chunks.chunk_sum[old_chunk_ind] += self.ranks[old_val];
                sample_chunks.chunk_sum[new_chunk] -= self.ranks[new_val];

                let remove_pos = match sample_chunks.chunks[new_chunk].binary_search(&new_val) {
                    Ok(p) => p,
                    Err(_) => {
                        if has_cand {
                            has_cand = false;
                        }
                        continue;
                    }
                };
                sample_chunks.chunks[new_chunk].remove(remove_pos);
                sample_chunks.chunks[old_chunk_ind].insert(old_ind_in_chunk, old_val);

                if has_cand && new_val as isize == cand_val {
                    has_cand = false;
                }
                if has_cand {
                    if (old_val as i64) < cand_val as i64 {
                        cand_x -= 1;
                        cand_y += self.ranks[old_val];
                    }
                    if (new_val as i64) < cand_val as i64 {
                        cand_x += 1;
                        cand_y -= self.ranks[new_val];
                    }
                }
            } else {
                moves += 1;
            }
        }

        PerturbateResult { moves, iters }
    }
}
