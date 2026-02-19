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

    fn r_unif_index(&mut self, dn: u64) -> u64 {
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

    pub fn sample_int_one(&mut self, n: usize) -> usize {
        (self.r_unif_index(n as u64) + 1) as usize
    }

    pub fn consume_sample_shuffle(&mut self, n: usize) {
        // Mirrors R's sample(1:n) uniform no-replacement path in do_sample():
        // draws R_unif_index(dn) for dn = n, n-1, ..., 1.
        for dn in (1..=n).rev() {
            let _ = self.r_unif_index(dn as u64);
        }
    }
}

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
pub fn combination(a: usize, b: usize, k: usize, rng: &mut Mt19937Compat) -> Vec<usize> {
    let n = b - a + 1;
    let mut v = Vec::with_capacity(k);
    let mut used = vec![false; n];

    if (k as f64) < (n as f64) / 2.0 {
        for _ in 0..k {
            loop {
                let x = uid_wrapper(a, b, rng);
                let idx = x - a;
                if !used[idx] {
                    used[idx] = true;
                    v.push(x);
                    break;
                }
            }
        }
    } else {
        for r in (n - k)..n {
            let x = uid_wrapper(0, r, rng);
            if !used[x] {
                used[x] = true;
                v.push(a + x);
            } else {
                used[r] = true;
                v.push(a + r);
            }
        }
        // Fisher-Yates shuffle implemented explicitly for platform reproducibility.
        for i in (1..v.len()).rev() {
            let j = uid_wrapper(0, i, rng);
            v.swap(i, j);
        }
    }

    v
}

pub fn sample_int_one(n: usize, rng: &mut Mt19937Compat) -> usize {
    // Integer draw in [1, n].
    uid_wrapper(1, n, rng)
}

#[cfg(test)]
mod tests {
    use super::RMt19937SeedCompat;

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
}
