#![allow(dead_code)]

// FGSEA-compatible RNG helpers:
// - MT19937 engine compatible with boost::mt19937
// - uid_wrapper rejection sampling
// - combination() sampling used by fgsea C++ core

pub struct Mt19937Compat {
    mt: [u32; 624],
    idx: usize,
}

pub struct RMt19937SeedCompat {
    mt: [u32; 624],
    mti: usize,
}

pub struct RLecuyerCmrgSeedCompat {
    s: [u32; 6],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RSampleKind {
    Rounding,
    #[default]
    Rejection,
}

impl Mt19937Compat {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, idx: 624 }
    }

    fn twist(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908B0DF;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7FFF_FFFF;

        for i in 0..N {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
            let mut xa = x >> 1;
            if (x & 1) != 0 {
                xa ^= MATRIX_A;
            }
            self.mt[i] = self.mt[(i + M) % N] ^ xa;
        }
        self.idx = 0;
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    pub fn next_unif(&mut self) -> f64 {
        // Approximation of R's unif generation from 32-bit integer stream.
        (self.next_u32() as f64) / (u32::MAX as f64 + 1.0)
    }
}

impl RMt19937SeedCompat {
    pub fn from_r_set_seed(seed: u32) -> Self {
        // Mirrors R's RNG_Init(MERSENNE_TWISTER, seed) path:
        // - 50 rounds of LCG scrambling
        // - fill 625 seed slots with the same LCG
        // - FixupSeeds(initial=1): i_seed[0] = 624
        let mut s = seed;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }

        let mut i_seed = [0u32; 625];
        for slot in &mut i_seed {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *slot = s;
        }
        i_seed[0] = 624;

        let mut mt = [0u32; 624];
        mt.copy_from_slice(&i_seed[1..]);
        Self { mt, mti: 624 }
    }

    fn fixup(x: f64) -> f64 {
        // Same as R's fixup(): avoid exact 0 and 1.
        const I2_32M1: f64 = 2.328_306_437_080_797e-10; // 1/(2^32 - 1)
        if x <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if (1.0 - x) <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        x
    }

    fn next_unif_r(&mut self) -> f64 {
        // R's MT_genrand() for Mersenne-Twister.
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_B0DF;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7FFF_FFFF;

        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk - (N - M)] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 0x1) as usize];
            self.mti = 0;
        }

        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;

        let u = (y as f64) * 2.328_306_436_538_696_3e-10; // 1/2^32
        Self::fixup(u)
    }

    fn rbits(&mut self, bits: i32) -> u64 {
        // R's rbits(): generate integer chunks from floor(unif_rand()*65536).
        let mut v: u64 = 0;
        let mut n = 0;
        while n <= bits {
            let v1 = (self.next_unif_r() * 65536.0).floor() as u64;
            v = 65536u64.wrapping_mul(v).wrapping_add(v1);
            n += 16;
        }
        if bits == 64 {
            v
        } else {
            let mask = (1u64 << bits) - 1;
            v & mask
        }
    }

    fn r_unif_index_rounding(&mut self, dn: u64) -> u64 {
        if dn == 0 {
            return 0;
        }
        let dv = (self.next_unif_r() * dn as f64).floor() as u64;
        dv.min(dn - 1)
    }

    fn r_unif_index_rejection(&mut self, dn: u64) -> u64 {
        if dn == 0 {
            return 0;
        }
        let bits = (dn as f64).log2().ceil() as i32;
        loop {
            let dv = self.rbits(bits);
            if dv < dn {
                return dv;
            }
        }
    }

    fn r_unif_index_with_kind(&mut self, dn: u64, sample_kind: RSampleKind) -> u64 {
        match sample_kind {
            RSampleKind::Rounding => self.r_unif_index_rounding(dn),
            RSampleKind::Rejection => self.r_unif_index_rejection(dn),
        }
    }

    pub fn sample_int_one(&mut self, n: usize) -> usize {
        self.sample_int_one_with_kind(n, RSampleKind::Rejection)
    }

    pub fn sample_int_one_with_kind(&mut self, n: usize, sample_kind: RSampleKind) -> usize {
        (self.r_unif_index_with_kind(n as u64, sample_kind) + 1) as usize
    }

    pub fn sample_int_no_replace(&mut self, n: usize, size: usize) -> Vec<usize> {
        self.sample_int_no_replace_with_kind(n, size, RSampleKind::Rejection)
    }

    pub fn sample_int_no_replace_with_kind(
        &mut self,
        n: usize,
        size: usize,
        sample_kind: RSampleKind,
    ) -> Vec<usize> {
        assert!(
            size <= n,
            "sample_int_no_replace: size ({size}) must be <= n ({n})"
        );
        let mut pool: Vec<usize> = (1..=n).collect();
        let mut out = Vec::with_capacity(size);

        // Mirrors R's do_sample() no-replacement path: at step i, draw
        // j in [0, n-i-1], output pool[j], then move pool[n-i-1] into j.
        for i in 0..size {
            let dn = n - i;
            let j = self.r_unif_index_with_kind(dn as u64, sample_kind) as usize;
            out.push(pool[j]);
            pool[j] = pool[dn - 1];
        }

        out
    }

    pub fn consume_sample_shuffle(&mut self, n: usize) {
        self.consume_sample_shuffle_with_kind(n, RSampleKind::Rejection);
    }

    pub fn consume_sample_shuffle_with_kind(&mut self, n: usize, sample_kind: RSampleKind) {
        // Mirrors R's sample(1:n) uniform no-replacement path in do_sample():
        // draws R_unif_index(dn) for dn = n, n-1, ..., 1.
        for dn in (1..=n).rev() {
            let _ = self.r_unif_index_with_kind(dn as u64, sample_kind);
        }
    }
}

impl RLecuyerCmrgSeedCompat {
    pub fn from_r_set_seed(seed: u32) -> Self {
        // Mirrors R's RNG_Init(LECUYER_CMRG, seed) in RNG.c.
        const M2: u32 = 4_294_944_443;
        let mut s = seed;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        let mut state = [0u32; 6];
        for slot in &mut state {
            s = s.wrapping_mul(69069).wrapping_add(1);
            while s >= M2 {
                s = s.wrapping_mul(69069).wrapping_add(1);
            }
            *slot = s;
        }
        Self { s: state }
    }

    fn next_unif_r(&mut self) -> f64 {
        // Mirrors R's unif_rand() L'Ecuyer-CMRG branch in RNG.c.
        const M1: i64 = 4_294_967_087;
        const M2: i64 = 4_294_944_443;
        const NORMC: f64 = 2.328_306_549_295_728e-10;
        const A12: i64 = 1_403_580;
        const A13N: i64 = 810_728;
        const A21: i64 = 527_612;
        const A23N: i64 = 1_370_589;

        let mut p1 = A12 * self.s[1] as i64 - A13N * self.s[0] as i64;
        let k1 = p1 / M1;
        p1 -= k1 * M1;
        if p1 < 0 {
            p1 += M1;
        }
        self.s[0] = self.s[1];
        self.s[1] = self.s[2];
        self.s[2] = p1 as u32;

        let mut p2 = A21 * self.s[5] as i64 - A23N * self.s[3] as i64;
        let k2 = p2 / M2;
        p2 -= k2 * M2;
        if p2 < 0 {
            p2 += M2;
        }
        self.s[3] = self.s[4];
        self.s[4] = self.s[5];
        self.s[5] = p2 as u32;

        let z = if p1 > p2 { p1 - p2 } else { p1 - p2 + M1 };
        (z as f64) * NORMC
    }

    fn rbits(&mut self, bits: i32) -> u64 {
        // Same integer extraction used by R_unif_index.
        let mut v: u64 = 0;
        let mut n = 0;
        while n <= bits {
            let v1 = (self.next_unif_r() * 65536.0).floor() as u64;
            v = 65536u64.wrapping_mul(v).wrapping_add(v1);
            n += 16;
        }
        if bits == 64 {
            v
        } else {
            let mask = (1u64 << bits) - 1;
            v & mask
        }
    }

    fn r_unif_index_rounding(&mut self, dn: u64) -> u64 {
        if dn == 0 {
            return 0;
        }
        let dv = (self.next_unif_r() * dn as f64).floor() as u64;
        dv.min(dn - 1)
    }

    fn r_unif_index_rejection(&mut self, dn: u64) -> u64 {
        if dn == 0 {
            return 0;
        }
        let bits = (dn as f64).log2().ceil() as i32;
        loop {
            let dv = self.rbits(bits);
            if dv < dn {
                return dv;
            }
        }
    }

    fn r_unif_index_with_kind(&mut self, dn: u64, sample_kind: RSampleKind) -> u64 {
        match sample_kind {
            RSampleKind::Rounding => self.r_unif_index_rounding(dn),
            RSampleKind::Rejection => self.r_unif_index_rejection(dn),
        }
    }

    pub fn sample_int_no_replace(&mut self, n: usize, size: usize) -> Vec<usize> {
        self.sample_int_no_replace_with_kind(n, size, RSampleKind::Rejection)
    }

    pub fn sample_int_no_replace_with_kind(
        &mut self,
        n: usize,
        size: usize,
        sample_kind: RSampleKind,
    ) -> Vec<usize> {
        assert!(
            size <= n,
            "sample_int_no_replace: size ({size}) must be <= n ({n})"
        );
        let mut pool: Vec<usize> = (1..=n).collect();
        let mut out = Vec::with_capacity(size);

        for i in 0..size {
            let dn = n - i;
            let j = self.r_unif_index_with_kind(dn as u64, sample_kind) as usize;
            out.push(pool[j]);
            pool[j] = pool[dn - 1];
        }

        out
    }
}

#[inline(always)]
pub fn uid_wrapper(from: usize, to: usize, rng: &mut Mt19937Compat) -> usize {
    let len = to - from + 1;
    let max_val = u32::MAX;
    let complete_part = max_val - (max_val % (len as u32));
    loop {
        let x = rng.next_u32();
        if x < complete_part {
            return from + (x % (len as u32)) as usize;
        }
    }
}

// Generates k unique integers in [a, b] (inclusive), preserving fgsea util.cpp behavior.
// Uses a stamp-based marker array to avoid per-call allocation of `used`.
pub fn combination(a: usize, b: usize, k: usize, rng: &mut Mt19937Compat) -> Vec<usize> {
    let n = b - a + 1;
    let mut v = Vec::with_capacity(k);

    // Thread-local stamp array to avoid allocating `used` every call.
    thread_local!(static COMB_MARKER: std::cell::RefCell<Option<(Vec<u32>, u32)>> = const { std::cell::RefCell::new(None) });

    COMB_MARKER.with(|marker| {
        let mut guard = marker.borrow_mut();
        let (marks, stamp_ref) = guard.get_or_insert_with(|| (vec![0u32; n.max(10000)], 0u32));

        // Resize if needed (shouldn't happen after first real call).
        if marks.len() < n {
            marks.resize(n, 0);
        }

        // Increment stamp; reset if overflow.
        let mut stamp = stamp_ref.wrapping_add(1);
        if stamp == 0 {
            marks.fill(0);
            stamp = 1;
        }
        *stamp_ref = stamp;

        if (k as f64) < (n as f64) / 2.0 {
            for _ in 0..k {
                loop {
                    let x = uid_wrapper(a, b, rng);
                    let idx = x - a;
                    if marks[idx] != stamp {
                        marks[idx] = stamp;
                        v.push(x);
                        break;
                    }
                }
            }
        } else {
            for r in (n - k)..n {
                let x = uid_wrapper(0, r, rng);
                if marks[x] != stamp {
                    marks[x] = stamp;
                    v.push(a + x);
                } else {
                    marks[r] = stamp;
                    v.push(a + r);
                }
            }
            // Fisher-Yates shuffle implemented explicitly for platform reproducibility.
            for i in (1..v.len()).rev() {
                let j = uid_wrapper(0, i, rng);
                v.swap(i, j);
            }
        }
    });

    v
}

pub fn sample_int_one(n: usize, rng: &mut Mt19937Compat) -> usize {
    // Integer draw in [1, n].
    uid_wrapper(1, n, rng)
}

#[cfg(test)]
mod tests {
    use super::{RLecuyerCmrgSeedCompat, RMt19937SeedCompat, RSampleKind};

    #[test]
    fn r_seed_sequence_matches_sample_int_onee9_reference() {
        let cases = [
            (1u32, 66_608_964usize, 312_928_385usize),
            (42u32, 707_850_213usize, 155_243_673usize),
            (123u32, 161_401_295usize, 682_811_918usize),
            (999u32, 597_333_316usize, 406_641_223usize),
            (2026u32, 853_315_193usize, 601_901_350usize),
        ];
        for (seed, expected1, expected2) in cases {
            let mut r = RMt19937SeedCompat::from_r_set_seed(seed);
            let got1 = r.sample_int_one(1_000_000_000);
            let got2 = r.sample_int_one(1_000_000_000);
            assert_eq!(got1, expected1, "seed={seed}, first draw mismatch");
            assert_eq!(got2, expected2, "seed={seed}, second draw mismatch");
        }
    }

    #[test]
    fn r_seed_sequence_matches_fgsea_multilevel_group_order_then_seed() {
        // Reference from traced fgseaMultilevel run with set.seed(42), where:
        // 1) simple seed <- sample.int(1e9, 1) = 707850213
        // 2) indxs <- sample(1:2) consumes shuffle RNG draws
        // 3) multilevel seed <- sample.int(1e9, 1) = 608797924
        let mut r = RMt19937SeedCompat::from_r_set_seed(42);
        let simple_seed = r.sample_int_one(1_000_000_000);
        assert_eq!(simple_seed, 707_850_213usize);
        r.consume_sample_shuffle(2);
        let multilevel_seed = r.sample_int_one(1_000_000_000);
        assert_eq!(multilevel_seed, 608_797_924usize);
    }

    #[test]
    fn rounding_sample_kind_matches_r_reference() {
        // Reference generated in R 4.4.3 with:
        // RNGkind("Mersenne-Twister", "Inversion", "Rounding")
        // set.seed(42); sample.int(1e9, 1)
        // set.seed(20260322); sample.int(1e9, 1); sample.int(20, 5)
        let mut seed42 = RMt19937SeedCompat::from_r_set_seed(42u32);
        assert_eq!(
            seed42.sample_int_one_with_kind(1_000_000_000, RSampleKind::Rounding),
            914_806_044usize
        );

        let mut seed20260322 = RMt19937SeedCompat::from_r_set_seed(20_260_322u32);
        assert_eq!(
            seed20260322.sample_int_one_with_kind(1_000_000_000, RSampleKind::Rounding),
            367_440_000usize
        );
        let mut seed20260322_fresh = RMt19937SeedCompat::from_r_set_seed(20_260_322u32);
        assert_eq!(
            seed20260322_fresh.sample_int_no_replace_with_kind(20, 5, RSampleKind::Rounding),
            vec![8usize, 9, 11, 17, 1]
        );
    }

    #[test]
    fn mt_sample_int_no_replace_matches_r_reference() {
        // Reference generated in R:
        // set.seed(707850213); sample.int(500, 14)
        let mut r = RMt19937SeedCompat::from_r_set_seed(707_850_213u32);
        let got = r.sample_int_no_replace(500, 14);
        let expected = vec![
            427usize, 52, 329, 411, 196, 105, 4, 358, 27, 442, 164, 400, 429, 206,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn mt_sample_int_no_replace_matches_small_r_reference() {
        // Reference generated in R:
        // set.seed(707850213); sample.int(20, 5); sample.int(20, 5)
        let mut r = RMt19937SeedCompat::from_r_set_seed(707_850_213u32);
        let got1 = r.sample_int_no_replace(20, 5);
        let got2 = r.sample_int_no_replace(20, 5);
        assert_eq!(got1, vec![11usize, 9, 4, 19, 18]);
        assert_eq!(got2, vec![6usize, 4, 16, 13, 14]);
    }

    #[test]
    fn lecuyer_sample_int_no_replace_matches_r_reference() {
        // Reference generated in R:
        // RNGkind("L'Ecuyer-CMRG"); set.seed(707850213); sample.int(500, 14)
        let mut r = RLecuyerCmrgSeedCompat::from_r_set_seed(707_850_213u32);
        let got = r.sample_int_no_replace(500, 14);
        let expected = vec![
            440usize, 272, 418, 57, 391, 245, 150, 93, 488, 293, 232, 163, 212, 378,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn lecuyer_rounding_sample_kind_matches_r_reference() {
        // Reference generated in R 4.4.3 with:
        // RNGkind("L'Ecuyer-CMRG", "Inversion", "Rounding")
        // set.seed(707850213); sample.int(500, 14)
        let mut r = RLecuyerCmrgSeedCompat::from_r_set_seed(707_850_213u32);
        let got = r.sample_int_no_replace_with_kind(500, 14, RSampleKind::Rounding);
        let expected = vec![
            156usize, 303, 190, 94, 372, 153, 461, 124, 77, 72, 90, 349, 311, 281,
        ];
        assert_eq!(got, expected);
    }
}
