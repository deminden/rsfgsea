use std::collections::HashSet;

// These helpers intentionally mirror CPython/NumPy behavior for blitz reference parity.
// Pathway ordering, random draws, and floating-point reductions are part of the mode contract.
#[derive(Clone)]
pub(super) struct PythonStringSet {
    table: Vec<Option<(u64, String)>>,
    used: usize,
    fill: usize,
}

impl PythonStringSet {
    pub(super) fn new() -> Self {
        Self {
            table: vec![None; 8],
            used: 0,
            fill: 0,
        }
    }

    pub(super) fn from_iter(values: impl IntoIterator<Item = String>) -> Self {
        let mut set = Self::new();
        for value in values {
            set.insert(value);
        }
        set
    }

    pub(super) fn len(&self) -> usize {
        self.used
    }

    pub(super) fn members(&self) -> HashSet<String> {
        self.table
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(_, value)| value.clone()))
            .collect()
    }

    pub(super) fn iter_values(&self) -> impl Iterator<Item = &String> {
        self.table
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(_, value)| value))
    }

    pub(super) fn intersection_new_set_order(&self, other: &HashSet<String>) -> Vec<String> {
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
pub(super) struct PythonIntSet {
    table: Vec<Option<usize>>,
    resize_scratch: Vec<usize>,
    used: usize,
    fill: usize,
}

impl PythonIntSet {
    pub(super) fn new() -> Self {
        Self {
            table: vec![None; 8],
            resize_scratch: Vec::new(),
            used: 0,
            fill: 0,
        }
    }

    pub(super) fn from_iter(values: impl IntoIterator<Item = usize>) -> Self {
        let mut set = Self::new();
        for value in values {
            set.insert(value);
        }
        set.resize_scratch = Vec::new();
        set
    }

    pub(super) fn len(&self) -> usize {
        self.used
    }

    pub(super) fn contains(&self, value: usize) -> bool {
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

    pub(super) fn iter_values(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|slot| *slot)
    }

    pub(super) fn insert_filtered(
        &mut self,
        values: impl IntoIterator<Item = usize>,
        predicate: impl Fn(usize) -> bool,
    ) {
        self.reset_to_new_set();
        for value in values {
            if predicate(value) {
                self.insert(value);
            }
        }
    }

    fn reset_to_new_set(&mut self) {
        if self.table.len() > 8 {
            self.table.truncate(8);
        } else if self.table.len() < 8 {
            self.table.resize(8, None);
        }
        self.table.fill(None);
        self.used = 0;
        self.fill = 0;
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
        let mut old = std::mem::take(&mut self.resize_scratch);
        old.clear();
        old.extend(self.table.iter_mut().filter_map(|slot| slot.take()));
        self.table.clear();
        self.table.resize(new_size, None);
        self.used = 0;
        self.fill = 0;
        for &value in &old {
            self.insert_no_resize(value);
        }
        self.resize_scratch = old;
    }
}

pub(super) fn python_ascii_hash_seed0(value: &str) -> u64 {
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
pub(super) struct NumpyMt19937 {
    mt: [u32; 624],
    mti: usize,
}

impl NumpyMt19937 {
    pub(super) fn new(seed: u32) -> Self {
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
            let mt = self.mt.as_mut_ptr();
            unsafe {
                for kk in 0..(N - M) {
                    let y = (*mt.add(kk) & UPPER_MASK) | (*mt.add(kk + 1) & LOWER_MASK);
                    *mt.add(kk) =
                        *mt.add(kk + M) ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
                }
                for kk in (N - M)..(N - 1) {
                    let y = (*mt.add(kk) & UPPER_MASK) | (*mt.add(kk + 1) & LOWER_MASK);
                    *mt.add(kk) =
                        *mt.add(kk + M - N) ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
                }
                let y = (*mt.add(N - 1) & UPPER_MASK) | (*mt.add(0) & LOWER_MASK);
                *mt.add(N - 1) = *mt.add(M - 1) ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            }
            self.mti = 0;
        }

        let mut y = unsafe { *self.mt.get_unchecked(self.mti) };
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    #[cfg(test)]
    fn random_interval(&mut self, max: usize) -> usize {
        let mask = (max + 1).next_power_of_two() as u32 - 1;
        self.random_interval_with_mask(max, mask)
    }

    #[inline(always)]
    fn random_interval_with_mask(&mut self, max: usize, mask: u32) -> usize {
        loop {
            let value = self.next_u32() & mask;
            if value <= max as u32 {
                return value as usize;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn choice_without_replacement(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut values: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.random_interval(i);
            values.swap(i, j);
        }
        values.truncate(k);
        values
    }

    pub(super) fn choice_without_replacement_into<'a>(
        &mut self,
        n: usize,
        k: usize,
        values: &'a mut Vec<usize>,
    ) -> &'a [usize] {
        fill_usize_sequence(values, n);
        let mut mask = n.next_power_of_two() as u32 - 1;
        for i in (1..n).rev() {
            if i == (mask >> 1) as usize {
                mask = i as u32;
            }
            let j = self.random_interval_with_mask(i, mask);
            swap_indices(values, i, j);
        }
        values.truncate(k);
        values
    }

    pub(super) fn choice_without_replacement_u32_into<'a>(
        &mut self,
        n: u32,
        k: usize,
        values: &'a mut Vec<u32>,
    ) -> &'a [u32] {
        fill_u32_sequence(values, n);
        let n = values.len();
        let mut mask = n.next_power_of_two() as u32 - 1;
        for i in (1..n).rev() {
            if i == (mask >> 1) as usize {
                mask = i as u32;
            }
            let j = self.random_interval_with_mask(i, mask);
            swap_indices(values, i, j);
        }
        values.truncate(k);
        values
    }

    fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    pub(super) fn standard_normals(&mut self, n: usize) -> Vec<f64> {
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

#[inline(always)]
fn fill_usize_sequence(values: &mut Vec<usize>, n: usize) {
    values.clear();
    if values.capacity() < n {
        values.reserve_exact(n - values.capacity());
    }
    let spare = values.spare_capacity_mut();
    unsafe {
        for i in 0..n {
            spare.get_unchecked_mut(i).write(i);
        }
        values.set_len(n);
    }
}

#[inline(always)]
fn fill_u32_sequence(values: &mut Vec<u32>, n: u32) {
    let len = n as usize;
    values.clear();
    if values.capacity() < len {
        values.reserve_exact(len - values.capacity());
    }
    let spare = values.spare_capacity_mut();
    unsafe {
        for i in 0..len {
            spare.get_unchecked_mut(i).write(i as u32);
        }
        values.set_len(len);
    }
}

#[inline(always)]
fn swap_indices<T>(values: &mut [T], i: usize, j: usize) {
    debug_assert!(i < values.len());
    debug_assert!(j < values.len());
    unsafe {
        std::ptr::swap(values.as_mut_ptr().add(i), values.as_mut_ptr().add(j));
    }
}

#[allow(clippy::approx_constant, clippy::excessive_precision)]
pub(super) fn numpy_log_f32(x_in: f32) -> f32 {
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

pub(super) fn numpy_pairwise_sum_f32(values: &[f32]) -> f32 {
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

#[cfg(test)]
pub(super) fn python_int_set_iteration_order(values: &[usize]) -> Vec<usize> {
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

pub(super) fn numpy_pairwise_sum_f64(values: &[f64]) -> f64 {
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

pub(super) fn numpy_hit_score_sum_f64(values: &[f64]) -> f64 {
    if values.len() == 7 {
        (values[0] + (values[1] + values[2])) + ((values[3] + values[4]) + (values[5] + values[6]))
    } else {
        numpy_pairwise_sum_f64(values)
    }
}
